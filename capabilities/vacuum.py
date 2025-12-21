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
    
    PROMPT = {
        "system": """Extract vacuum command details from the user's request.

- mode: 'vacuum' for dry cleaning (saugen, staubsaugen), 'mop' for wet cleaning (wischen, feucht)
- area: Room name (without articles like 'den', 'die', 'das'), or null if not specified
- floor: Floor name (Erdgeschoss, Obergeschoss, Keller, etc.), or null if not specified
- scope: 'GLOBAL' if whole house/apartment mentioned, or null if not specified

Examples:
"Sauge die Küche" → {"mode": "vacuum", "area": "Küche", "floor": null, "scope": null}
"Wische den Keller" → {"mode": "mop", "area": "Keller", "floor": null, "scope": null}
"Staubsauge das Erdgeschoss" → {"mode": "vacuum", "area": null, "floor": "Erdgeschoss", "scope": null}
"Sauge das ganze Haus" → {"mode": "vacuum", "area": null, "floor": null, "scope": "GLOBAL"}""",
        "schema": {
            "properties": {
                "mode": {"type": "string"},
                "area": {"type": ["string", "null"]},
                "floor": {"type": ["string", "null"]},
                "scope": {"type": ["string", "null"]},
            }
        }
    }

    async def run(self, user_input, intent_name: str, slots: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        if intent_name != "HassVacuumStart":
            return {}

        # Extract vacuum details via LLM
        extracted = await self._extract_vacuum_details(user_input.text)
        
        mode = extracted.get("mode", "vacuum")
        scope = extracted.get("scope")
        floor_name = extracted.get("floor")
        area_name = extracted.get("area")

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
        from .area_resolver import AreaResolverCapability as AreaAliasCapability
        from homeassistant.helpers import area_registry as ar

        # Check for exact match in registry first to save LLM call
        registry = ar.async_get(self.hass)
        for a in registry.async_list_areas():
            if a.name.lower() == name.lower():
                return a.name

        # Ask LLM for alias
        alias_cap = AreaAliasCapability(self.hass, self.config)
        res = await alias_cap.run(user_input, search_text=name)
        mapped = res.get("area")
        
        if mapped and mapped != "GLOBAL":
            return mapped
            
        return name # Fallback to original if no mapping found

    async def _extract_vacuum_details(self, text: str) -> Dict[str, Any]:
        """Extract vacuum details using LLM."""
        try:
            result = await self._safe_prompt(
                self.PROMPT, {"user_input": text}, temperature=0.0
            )
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            _LOGGER.debug(f"Failed to extract vacuum details: {e}")
        return {}