import logging
from typing import Any, Dict, List, Optional

from .base import Capability
from ..utils.response_builder import build_state_response

_LOGGER = logging.getLogger(__name__)


class IntentConfirmationCapability(Capability):
    """
    Generates a short, natural confirmation sentence using an LLM.
    Dynamically builds the prompt based on the executed intent.
    """

    name = "intent_confirmation"
    description = "Generates a natural language confirmation for an action."

    INTENT_DESCRIPTIONS = {
        "HassTurnOn": "The device(s) were turned ON.",
        "HassTurnOff": "The device(s) were turned OFF.",
        "HassLightSet": "Light settings were adjusted.",
        "HassSetPosition": "Cover/Blind position was set.",
        "HassClimateSetTemperature": "Thermostat target temperature was changed.",
        "HassTimerSet": "A timer was successfully set.",
        "HassGetState": "A state or measurement was queried.",
        "TemporaryControl": "The device was switched on/off TEMPORARILY.",
        "DelayedControl": "The action will be executed AFTER a delay.",
        "HassVacuumStart": "Vacuum/Mop was started.",
    }
    
    # Domain-specific overrides for certain intents
    DOMAIN_INTENT_DESCRIPTIONS = {
        "cover": {
            "HassTurnOn": "The cover(s)/blind(s) are being OPENED (werden geöffnet).",
            "HassTurnOff": "The cover(s)/blind(s) are being CLOSED (werden geschlossen).",
        },
    }

    SCHEMA = {"properties": {"response": {"type": "string"}}, "required": ["response"]}

    def _get_action_description(self, intent_name: str, domain: str) -> str:
        """Get domain-aware action description."""
        # Check for domain-specific override first
        if domain in self.DOMAIN_INTENT_DESCRIPTIONS:
            if intent_name in self.DOMAIN_INTENT_DESCRIPTIONS[domain]:
                return self.DOMAIN_INTENT_DESCRIPTIONS[domain][intent_name]
        # Fall back to generic description
        return self.INTENT_DESCRIPTIONS.get(intent_name, "An action was performed.")

    async def run(
        self,
        user_input,
        intent_name: str,
        entity_ids: List[str],
        params: Dict[str, Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:

        names = []
        domains = []
        states = []

        for eid in entity_ids:
            st = self.hass.states.get(eid)
            if st:
                names.append(st.attributes.get("friendly_name") or eid)
                domains.append(eid.split(".")[0])
                states.append(st.state)
            else:
                names.append(eid)
                domains.append(eid.split(".")[0] if "." in eid else "")
                states.append("unknown")

        # Template-based response for state queries (no LLM needed)
        if intent_name == "HassGetState":
            domain = domains[0] if domains else None
            message = build_state_response(names, states, domain)
            _LOGGER.debug("[IntentConfirmation] Template response for HassGetState: '%s'", message)
            return {"message": message}

        ignored_keys = {"domain", "service", "entity_id", "area_id"}
        relevant_params = {
            k: v for k, v in (params or {}).items() if k not in ignored_keys
        }

        # Get domain-aware action description
        primary_domain = domains[0] if domains else ""
        action_desc = self._get_action_description(intent_name, primary_domain)

        # Build base rules
        base_rules = """1. **Identify the Device:** Use the device names from 'devices' list directly. Don't substitute with area names.
2. **Duration:** ONLY mention duration if 'duration' is explicitly set in params. If duration is null or missing, do NOT mention any time.
3. **Use Future Tense for covers:** - CORRECT: "Rollladen wird geschlossen." - WRONG: "Rollladen ist geschlossen."
4. **NEVER INVENT:** Do not add information that is not in the params."""

        # Add brightness guidance only for HassLightSet
        if intent_name == "HassLightSet":
            base_rules += """
5. **Brightness (IMPORTANT - check 'direction' in params):** 
   - If 'brightness' has a NUMBER: say "ist auf [X]% gesetzt."
   - If 'direction' is "increased": say "[Licht] ist jetzt heller."
   - If 'direction' is "decreased": say "[Licht] ist jetzt dunkler."
   - NEVER guess direction from command - ONLY use the 'direction' field!"""

        # Add cover direction guidance for HassSetPosition
        if intent_name == "HassSetPosition" and primary_domain == "cover":
            base_rules += """
5. **Cover Position (IMPORTANT - check 'direction' in params):**
   - If 'direction' is "increased": say "[Rollladen] ist jetzt weiter geöffnet." or "wird geöffnet."
   - If 'direction' is "decreased": say "[Rollladen] ist jetzt weiter geschlossen." or "wird geschlossen."
   - If 'position' has a NUMBER: say "[Rollladen] ist auf [X]%."
   - NEVER guess direction - ONLY use the 'direction' field!"""

        system = f"""You are a smart home assistant.
Generate a VERY SHORT, natural German confirmation (du-form).

## Context
Action: {action_desc}

## Rules
{base_rules}
"""

        payload = {
            "intent": intent_name,
            "devices": names,
            "domains": domains,
            "states": states,
            "params": relevant_params,
        }

        data = await self._safe_prompt(
            {"system": system, "schema": self.SCHEMA}, payload
        )
        message = (
            data.get("response", "Aktion ausgeführt.")
            if isinstance(data, dict)
            else "Aktion ausgeführt."
        )

        _LOGGER.debug("[IntentConfirmation] Generated: '%s'", message)
        return {"message": message}

