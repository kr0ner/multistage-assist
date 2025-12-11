import logging
from typing import Any, Dict, Optional
from homeassistant.core import Context
from .base import Capability
from custom_components.multistage_assist.conversation_utils import make_response

_LOGGER = logging.getLogger(__name__)

class VacuumCapability(Capability):
    """
    Control vacuums via 'script.vacuum_universal_manager'.
    The script handles room/floor/global logic internally.
    """
    name = "vacuum"
    description = "Control vacuum robots."

    SCRIPT_ENTITY_ID = "script.vacuum_universal_clean"

    async def run(self, user_input, intent_name: str, slots: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        if intent_name != "HassVacuumStart":
            return {}

        mode = slots.get("mode", "vacuum")
        scope = slots.get("scope")
        floor_name = slots.get("floor")
        area_name = slots.get("area")

        target_val = None
        
        # 1. Global Scope ("Sauge das ganze Haus")
        if scope == "GLOBAL" or (area_name and area_name.lower() in ("haus", "wohnung", "alles", "ganze haus")):
            target_val = "Alles"

        # 2. Floor Scope ("Wische das Erdgeschoss")
        elif floor_name:
            # We pass the floor name directly to the script
            target_val = floor_name

        # 3. Room Scope ("Staubsauge die Küche")
        elif area_name:
            # We still resolve aliases (e.g. "Bad" -> "Badezimmer") to ensure the script finds the room
            normalized = await self._normalize_area_name(user_input, area_name)
            target_val = normalized if normalized else area_name

        if not target_val:
             return {
                "status": "handled",
                "result": await make_response("Ich habe kein Ziel (Raum oder Etage) verstanden.", user_input)
            }

        # 4. Execute Script
        _LOGGER.info("[Vacuum] Calling script for target='%s', mode='%s'", target_val, mode)
        
        try:
            await self.hass.services.async_call(
                "script", "turn_on",
                {
                    "entity_id": self.SCRIPT_ENTITY_ID,
                    "variables": {"target": target_val, "mode": mode}
                },
                context=Context(),
                blocking=False
            )
        except Exception as e:
            _LOGGER.error("Failed to trigger vacuum script: %s", e)
            return {
                "status": "handled",
                "result": await make_response("Fehler beim Starten des Saugroboters.", user_input)
            }

        # 5. Confirmation
        action = "wischen" if mode == "mop" else "saugen"
        msg_target = "das Haus" if target_val == "Alles" else target_val
        
        return {
            "status": "handled",
            "result": await make_response(f"Alles klar, ich lasse {msg_target} {action}.", user_input)
        }

    async def _normalize_area_name(self, user_input, name: str) -> Optional[str]:
        """
        Use AreaAliasCapability to normalize 'Bad' -> 'Badezimmer'.
        This ensures the script receives the correct HA area name.
        """
        from .area_alias import AreaAliasCapability
        from homeassistant.helpers import area_registry as ar

        # Check for exact match in registry first to save LLM call
        registry = ar.async_get(self.hass)
        for a in registry.async_list_areas():
            if a.name.lower() == name.lower():
                return a.name

        # Ask LLM for alias
        alias_cap = AreaAliasCapability(self.hass, {})
        res = await alias_cap.run(user_input, search_text=name)
        mapped = res.get("area")
        
        if mapped and mapped != "GLOBAL":
            return mapped
            
        return name # Fallback to original if no mapping found