import logging
from typing import Any, Dict, List, Optional

from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentConfirmationCapability(Capability):
    """
    Generates a short, natural confirmation sentence using an LLM.
    Replaces static "Okay" responses.
    """

    name = "intent_confirmation"
    description = "Generates a natural language confirmation for an action."

    PROMPT = {
        "system": """You are a smart home assistant.
Generate a VERY SHORT, natural German confirmation for the action described.
Do not say "Okay" or "Erledigt". Just describe the new state.

## Rules
1. Use the 'domains' field to understand what the device is.
   - domain="light" + name="Küche" -> "Das Licht in der Küche" (NOT "Die Spülmaschine").
   - domain="cover" + name="Wohnzimmer" -> "Der Rollladen im Wohnzimmer".
   - domain="climate" -> "Die Heizung/Thermostat".
2. If the name is generic (e.g. "Küche"), combine it with the domain description.
3. Keep it brief.

## Examples
Input: {"intent": "HassTurnOn", "devices": ["Küche"], "domains": ["light"], "params": {}}
Output: {"response": "Das Licht in der Küche ist an."}

Input: {"intent": "HassTurnOff", "devices": ["Büro"], "domains": ["cover"], "params": {}}
Output: {"response": "Der Rollladen im Büro ist geschlossen."}

Input: {"intent": "HassLightSet", "devices": ["Wohnzimmer"], "domains": ["light"], "params": {"brightness": 50}}
Output: {"response": "Licht im Wohnzimmer auf 50% gestellt."}

## Input Format
Input JSON: {"intent": "...", "devices": ["..."], "domains": ["..."], "params": {...}}
Output JSON: {"response": "string"}
""",
        "schema": {
            "properties": {
                "response": {"type": "string"}
            },
            "required": ["response"]
        }
    }

    async def run(
        self,
        user_input,
        intent_name: str,
        entity_ids: List[str],
        params: Dict[str, Any] = None,
        **_: Any
    ) -> Dict[str, Any]:
        
        # 1. Resolve Friendly Names
        names = []
        for eid in entity_ids:
            st = self.hass.states.get(eid)
            if st:
                names.append(st.attributes.get("friendly_name") or eid)
            else:
                names.append(eid)

        # 2. Extract Domains (Critical for context)
        # e.g. ['light', 'switch']
        domains = list({eid.split(".")[0] for eid in entity_ids})
        
        # 3. Filter Parameters (Feed only what matters for speech)
        ignored_keys = {"domain", "service", "entity_id", "area_id"}
        relevant_params = {k: v for k, v in (params or {}).items() if k not in ignored_keys}

        payload = {
            "intent": intent_name,
            "devices": names,
            "domains": domains,  # <--- Added Domain Context
            "params": relevant_params
        }

        # 4. Generate
        data = await self._safe_prompt(self.PROMPT, payload)
        
        message = "Aktion ausgeführt."
        if isinstance(data, dict):
            message = data.get("response", message)

        _LOGGER.debug("[IntentConfirmation] Generated: '%s'", message)
        return {"message": message}