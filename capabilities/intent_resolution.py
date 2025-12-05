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

    def __init__(self, hass, config):
        super().__init__(hass, config)
        # Instantiate sub-capabilities
        self.keyword_cap = KeywordIntentCapability(hass, config)
        self.resolver_cap = EntityResolverCapability(hass, config)
        self.alias_cap = AreaAliasCapability(hass, config)
        self.memory_cap = MemoryCapability(hass, config)

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        """
        Returns:
        {
            "intent": str,
            "slots": dict,
            "entity_ids": list,
            "learning_data": dict (optional, if new alias found to learn)
        }
        or {} if failed.
        """
        
        # 1. Identify Intent & Slots (Keyword)
        ki_data = await self.keyword_cap.run(user_input)
        intent_name = ki_data.get("intent")
        slots = ki_data.get("slots") or {}

        if not intent_name:
            _LOGGER.debug("[IntentResolution] No intent found.")
            return {}

        entity_ids = []
        name_slot = slots.get("name")
        
        # 2. CHECK MEMORY FOR ENTITY ALIAS (Fast Path)
        # If the user named a device specifically ("Spiegellicht"), check if we learned it.
        if name_slot:
            known_eid = await self.memory_cap.get_entity_alias(name_slot)
            if known_eid:
                _LOGGER.debug("[IntentResolution] Memory hit! Entity '%s' -> %s", name_slot, known_eid)
                # Verify it still exists in HA
                if self.hass.states.get(known_eid):
                    entity_ids = [known_eid]

        # 3. Resolve Entities (Standard)
        if not entity_ids:
            er_data = await self.resolver_cap.run(user_input, entities=slots)
            entity_ids = er_data.get("resolved_ids") or []

        learning_data = None

        # 4. Fallback: Area Alias (Memory -> LLM)
        if not entity_ids:
            candidate_area = slots.get("area") or slots.get("name")
            
            if candidate_area:
                # A. Check Memory for Area
                mapped_area = await self.memory_cap.get_area_alias(candidate_area)
                is_new = False
                
                # B. Check LLM if not in memory
                if not mapped_area:
                    alias_res = await self.alias_cap.run(user_input, search_text=candidate_area)
                    mapped_area = alias_res.get("area")
                    if mapped_area:
                        is_new = True

                if mapped_area:
                    # Apply mapping
                    new_slots = slots.copy()
                    
                    if mapped_area == "GLOBAL":
                        _LOGGER.debug("[IntentResolution] Global scope detected via alias '%s'.", candidate_area)
                        new_slots.pop("area", None)
                        # Remove name if it was just the area name (e.g. "Schalte Haus aus")
                        if new_slots.get("name") == candidate_area:
                            new_slots.pop("name")
                    else:
                        _LOGGER.debug("[IntentResolution] Mapping area '%s' -> '%s'", candidate_area, mapped_area)
                        new_slots["area"] = mapped_area
                        if new_slots.get("name") == candidate_area:
                            new_slots.pop("name")
                    
                    # Retry Resolution with mapped area
                    er_data = await self.resolver_cap.run(user_input, entities=new_slots)
                    entity_ids = er_data.get("resolved_ids") or []
                    
                    # Prepare Learning Data if successful, new, and not a global keyword
                    if entity_ids and is_new and mapped_area != "GLOBAL":
                         learning_data = {
                             "type": "area",
                             "source": candidate_area,
                             "target": mapped_area
                         }
                    
                    # Update slots for execution params
                    slots = new_slots

        if not entity_ids:
            _LOGGER.debug("[IntentResolution] Failed to resolve entities for intent %s", intent_name)
            return {}

        return {
            "intent": intent_name,
            "slots": slots,
            "entity_ids": entity_ids,
            "learning_data": learning_data
        }