import logging
from typing import Any, Dict, List, Optional

from .base import Capability
from .keyword_intent import KeywordIntentCapability
from .entity_resolver import EntityResolverCapability
from .area_alias import AreaAliasCapability
from .memory import MemoryCapability

_LOGGER = logging.getLogger(__name__)

class IntentResolutionCapability(Capability):
    """
    Orchestrates the resolution of a single command string into an Intent + Entities.
    Combines KeywordIntent, EntityResolver, AreaAlias, and Memory logic.
    """

    name = "intent_resolution"
    description = "Resolves a command string to intent and entities."
    
    ENTITY_MATCH_PROMPT = {
        "system": """
You are a smart home helper.
User said a specific device name (e.g. "Spiegellicht") inside a specific area.
Map the user's name to one of the available entities in that area.

## Input
- user_name: The name the user said.
- candidates: List of available entities in the area (id, friendly_name).

## Rules
1. Find the best match based on meaning (e.g. "Spiegellicht" -> "Badezimmer Spiegel").
2. If valid match found, return the entity_id.
3. If no plausible match, return null.

## Output (STRICT)
JSON: {"entity_id": <string or null>}
""",
        "schema": {
            "properties": {"entity_id": {"type": ["string", "null"]}},
            "required": ["entity_id"]
        }
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.keyword_cap = KeywordIntentCapability(hass, config)
        self.resolver_cap = EntityResolverCapability(hass, config)
        self.alias_cap = AreaAliasCapability(hass, config)
        self.memory_cap = MemoryCapability(hass, config)

    async def _resolve_alias(self, user_input, text: str, mode: str) -> tuple[Optional[str], bool]:
        """Helper to resolve Area or Floor alias using Memory -> LLM."""
        if not text: return None, False
        
        # 1. Memory
        if mode == "floor":
            mapped = await self.memory_cap.get_floor_alias(text)
        else:
            mapped = await self.memory_cap.get_area_alias(text)
            
        if mapped:
            _LOGGER.debug("[IntentResolution] Memory hit (%s): '%s' -> '%s'", mode, text, mapped)
            return mapped, False

        # 2. LLM
        res = await self.alias_cap.run(user_input, search_text=text, mode=mode)
        mapped = res.get("match")
        
        if mapped:
            return mapped, True
            
        return None, False

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        ki_data = await self.keyword_cap.run(user_input)
        intent_name = ki_data.get("intent")
        slots = ki_data.get("slots") or {}

        if not intent_name:
            return {}

        entity_ids = []
        name_slot = slots.get("name")
        area_slot = slots.get("area")
        floor_slot = slots.get("floor")
        
        learning_data = None
        new_slots = slots.copy()

        # 1. Resolve FLOOR
        if floor_slot:
            mapped_floor, is_new_floor = await self._resolve_alias(user_input, floor_slot, "floor")
            if mapped_floor:
                new_slots["floor"] = mapped_floor
                # Don't learn if target is substring of source (e.g. "im Erdgeschoss" -> "Erdgeschoss")
                if is_new_floor and mapped_floor.lower() not in floor_slot.lower():
                    learning_data = {"type": "floor", "source": floor_slot, "target": mapped_floor}
        
        # 2. Resolve AREA
        if area_slot and not learning_data:
            mapped_area, is_new_area = await self._resolve_alias(user_input, area_slot, "area")
            if mapped_area:
                if mapped_area == "GLOBAL":
                    new_slots.pop("area", None)
                    if new_slots.get("name") == area_slot: new_slots.pop("name")
                else:
                    new_slots["area"] = mapped_area
                    if new_slots.get("name") == area_slot: new_slots.pop("name")
                
                # Don't learn if target is substring of source (e.g. "im Büro" -> "Büro")
                if is_new_area and mapped_area != "GLOBAL":
                    if mapped_area.lower() not in area_slot.lower():
                        learning_data = {"type": "area", "source": area_slot, "target": mapped_area}

        # 3. Check Entity Memory (Fast Path)
        if name_slot:
            known_eid = await self.memory_cap.get_entity_alias(name_slot)
            if known_eid and self.hass.states.get(known_eid):
                 entity_ids = [known_eid]

        # 4. Standard Resolution
        if not entity_ids:
            er_data = await self.resolver_cap.run(user_input, entities=new_slots)
            entity_ids = er_data.get("resolved_ids") or []

        # 5. Entity Alias Fallback
        if not entity_ids and name_slot and new_slots.get("area"):
             target_area = new_slots["area"]
             domain = new_slots.get("domain")
             area_candidates = self.resolver_cap._entities_in_area_by_name(target_area, domain)
             
             if area_candidates:
                 payload = {"user_name": name_slot, "candidates": area_candidates}
                 match_res = await self._safe_prompt(self.ENTITY_MATCH_PROMPT, payload)
                 matched_eid = match_res.get("entity_id")
                 if matched_eid and self.hass.states.get(matched_eid):
                     entity_ids = [matched_eid]
                     if not learning_data:
                         learning_data = {"type": "entity", "source": name_slot, "target": matched_eid}

        if not entity_ids:
            return {}

        return {
            "intent": intent_name,
            "slots": new_slots,
            "entity_ids": entity_ids,
            "learning_data": learning_data
        }