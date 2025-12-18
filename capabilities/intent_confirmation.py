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
        "HassTemporaryControl": "The device was switched on/off TEMPORARILY.",
        "HassVacuumStart": "Vacuum/Mop was started.",
    }

    SCHEMA = {"properties": {"response": {"type": "string"}}, "required": ["response"]}

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

        action_desc = self.INTENT_DESCRIPTIONS.get(
            intent_name, "An action was performed."
        )

        # Build base rules
        base_rules = """1. **Identify the Device:** Use the device names from 'devices' list directly. Don't substitute with area names.
2. **Duration:** ONLY mention duration if 'duration' is explicitly set in params. If duration is null or missing, do NOT mention any time.
3. **Use Future Tense for covers:** - CORRECT: "Rollladen wird geschlossen." - WRONG: "Rollladen ist geschlossen."
4. **NEVER INVENT:** Do not add information that is not in the params."""

        # Add brightness guidance only for HassLightSet
        if intent_name == "HassLightSet":
            base_rules += """
5. **Brightness:** 
   - If 'brightness' has a NUMBER: say "ist auf [X]% gesetzt."
   - If 'command' is "step_up" that means "[Licht] ist jetzt heller."
   - If 'command' is "step_down" that means "[Licht] ist jetzt dunkler." """

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

