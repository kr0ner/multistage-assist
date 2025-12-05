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
Example: "Das Licht im Bad ist an." or "Heizung ist auf 22 Grad."
Input JSON: {"intent": "...", "devices": ["..."], "params": {...}}
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
        
        # 2. Filter Parameters (Feed only what matters for speech)
        ignored_keys = {"domain", "service", "entity_id", "area_id"}
        relevant_params = {k: v for k, v in (params or {}).items() if k not in ignored_keys}

        payload = {
            "intent": intent_name,
            "devices": names,
            "params": relevant_params
        }

        # 3. Generate
        data = await self._safe_prompt(self.PROMPT, payload)
        
        message = "Aktion ausgeführt."
        if isinstance(data, dict):
            message = data.get("response", message)

        _LOGGER.debug("[IntentConfirmation] Generated: '%s'", message)
        return {"message": message}