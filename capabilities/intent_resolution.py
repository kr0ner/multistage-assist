import logging
from typing import Any, Dict, List, Optional

from .base import Capability
from .keyword_intent import KeywordIntentCapability
from .entity_resolver import EntityResolverCapability
from .area_resolver import AreaResolverCapability as AreaAliasCapability
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
            "required": ["entity_id"],
        },
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.keyword_cap = KeywordIntentCapability(hass, config)
        self.resolver_cap = EntityResolverCapability(hass, config)
        self.alias_cap = AreaAliasCapability(hass, config)
        self.memory_cap = None  # Will be injected by Stage1

    def set_memory(self, memory_cap):
        """Inject memory capability and propagate to resolver"""
        self.memory_cap = memory_cap
        # Also inject into resolver
        self.resolver_cap.set_memory(memory_cap)

    async def _resolve_alias(
        self, user_input, text: str, mode: str
    ) -> tuple[Optional[str], bool]:
        """Helper to resolve Area or Floor alias using Memory -> LLM."""
        if not text:
            return None, False

        # 1. Memory
        if mode == "floor":
            mapped = await self.memory_cap.get_floor_alias(text)
        else:
            mapped = await self.memory_cap.get_area_alias(text)

        if mapped:
            _LOGGER.debug(
                "[IntentResolution] Memory hit (%s): '%s' -> '%s'", mode, text, mapped
            )
            return mapped, False

        # 2. LLM
        res = await self.alias_cap.run(user_input, search_text=text, mode=mode)
        mapped = res.get("match")

        if mapped:
            return mapped, True

        return None, False

    async def run(self, user_input, ki_data: Dict[str, Any] = None, **_: Any) -> Dict[str, Any]:
        # Use provided ki_data or compute it (avoids duplicate LLM call)
        if ki_data is None:
            ki_data = await self.keyword_cap.run(user_input)
        
        intent_name = ki_data.get("intent")
        slots = ki_data.get("slots") or {}
        # IMPORTANT: keyword_intent returns domain at top level, not in slots
        detected_domain = ki_data.get("domain")

        if not intent_name:
            return {}

        entity_ids = []
        name_slot = slots.get("name")
        area_slot = slots.get("area")
        floor_slot = slots.get("floor")

        learning_data = None
        new_slots = slots.copy()
        
        # Inject the detected domain into slots for entity_resolver
        # This ensures area-based lookups filter by the correct entity type
        if detected_domain and not new_slots.get("domain"):
            new_slots["domain"] = detected_domain
            _LOGGER.debug(
                "[IntentResolution] Injecting domain '%s' from keyword_intent",
                detected_domain,
            )

        # --- 1. RECOVERY: Check for missed Area in raw text ---
        # If no area/floor extracted AND no specific name given, scan text for area
        # Skip this if name_slot is present - we'll resolve the entity by name directly
        if not area_slot and not floor_slot and not name_slot:
            # Check for global scope keywords - no need for LLM call
            from ..constants.messages_de import GLOBAL_KEYWORDS
            text_lower = user_input.text.lower()
            is_global_scope = any(kw in text_lower for kw in GLOBAL_KEYWORDS)
            
            if is_global_scope:
                _LOGGER.debug("[IntentResolution] Global scope detected, skipping area_alias LLM")
                # Don't set area - entity resolver will fetch all entities for the domain
            else:
                # Ask AreaAlias to scan the FULL text
                _LOGGER.debug(
                    "[IntentResolution] No area/name slot. Scanning full text for area..."
                )
                mapped_area, is_new = await self._resolve_alias(
                    user_input, user_input.text, "area"
                )

                if mapped_area and mapped_area != "GLOBAL":
                    _LOGGER.debug(
                        "[IntentResolution] Recovered area from text: %s", mapped_area
                    )
                    new_slots["area"] = mapped_area
                    area_slot = mapped_area  # Update local var for next steps
                    # We don't trigger learning here usually because it might be a fuzzy match on the whole sentence,
                    # but if it mapped a specific substring effectively, it's good.
        # -----------------------------------------------------

        # 2. Resolve FLOOR
        if floor_slot:
            mapped_floor, is_new_floor = await self._resolve_alias(
                user_input, floor_slot, "floor"
            )
            if mapped_floor:
                new_slots["floor"] = mapped_floor
                # Don't learn if target is substring of source (e.g. "im Erdgeschoss" -> "Erdgeschoss")
                if is_new_floor and mapped_floor.lower() not in floor_slot.lower():
                    learning_data = {
                        "type": "floor",
                        "source": floor_slot,
                        "target": mapped_floor,
                    }

        # 3. Resolve AREA
        if area_slot and not learning_data:
            mapped_area, is_new_area = await self._resolve_alias(
                user_input, area_slot, "area"
            )
            if mapped_area:
                if mapped_area == "GLOBAL":
                    new_slots.pop("area", None)
                    if new_slots.get("name") == area_slot:
                        new_slots.pop("name")
                else:
                    new_slots["area"] = mapped_area
                    if new_slots.get("name") == area_slot:
                        new_slots.pop("name")

                # Learn only if significant difference
                if is_new_area and mapped_area != "GLOBAL":
                    if mapped_area.lower() not in area_slot.lower():
                        learning_data = {
                            "type": "area",
                            "source": area_slot,
                            "target": mapped_area,
                        }

        # 4. Check Entity Memory
        if name_slot:
            known_eid = await self.memory_cap.get_entity_alias(name_slot)
            if known_eid and self.hass.states.get(known_eid):
                entity_ids = [known_eid]

        # 5. Standard Resolution
        if not entity_ids:
            er_data = await self.resolver_cap.run(user_input, entities=new_slots)
            entity_ids = er_data.get("resolved_ids") or []

        # 6. Entity Alias Fallback
        if not entity_ids and name_slot and new_slots.get("area"):
            target_area = new_slots["area"]
            domain = new_slots.get("domain")
            area_candidates = self.resolver_cap._entities_in_area_by_name(
                target_area, domain
            )

            if area_candidates:
                payload = {"user_name": name_slot, "candidates": area_candidates}
                match_res = await self._safe_prompt(self.ENTITY_MATCH_PROMPT, payload)
                matched_eid = match_res.get("entity_id")
                if matched_eid and self.hass.states.get(matched_eid):
                    entity_ids = [matched_eid]
                    if not learning_data:
                        learning_data = {
                            "type": "entity",
                            "source": name_slot,
                            "target": matched_eid,
                        }

        if not entity_ids:
            return {}

        return {
            "intent": intent_name,
            "slots": new_slots,
            "entity_ids": entity_ids,
            "learning_data": learning_data,
        }
