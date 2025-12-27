"""Intent confirmation using template-based responses.

No LLM dependency - uses randomized templates for natural variety.
"""

import logging
from typing import Any, Dict, List

from .base import Capability
from ..constants.messages_de import get_domain_confirmation, DOMAIN_RESPONSES
from ..utils.response_builder import build_state_response
from ..conversation_utils import join_names

_LOGGER = logging.getLogger(__name__)


class IntentConfirmationCapability(Capability):
    """
    Generates natural confirmation messages using templates.
    Each domain/intent has multiple variations for variety.
    """

    name = "intent_confirmation"
    description = "Generates a natural language confirmation for an action."

    async def run(
        self,
        user_input,
        intent_name: str,
        entity_ids: List[str],
        params: Dict[str, Any] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Generate confirmation message using templates."""
        params = params or {}
        
        # Gather entity info
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

        primary_domain = domains[0] if domains else "default"
        name_str = join_names(names)
        
        # Route to appropriate template
        action, value = self._get_action_and_value(intent_name, primary_domain, params, states)
        
        # Special case: state queries use build_state_response for multi-entity grouping
        if action == "_state_query":
            message = build_state_response(names, states, primary_domain)
            _LOGGER.debug("[IntentConfirmation] State query: '%s'", message)
            return {"message": message}
        
        message = get_domain_confirmation(
            domain=primary_domain,
            action=action,
            name=name_str,
            value=str(value) if value else "",
            area=params.get("area", ""),
            state=states[0] if states else "",
        )

        _LOGGER.debug("[IntentConfirmation] Template: domain=%s, action=%s -> '%s'", 
                      primary_domain, action, message)
        return {"message": message}


    def _get_action_and_value(
        self, 
        intent_name: str, 
        domain: str, 
        params: Dict[str, Any],
        states: List[str],
    ) -> tuple:
        """Determine action type and value based on intent and params."""
        
        # State queries - return special marker to use build_state_response
        if intent_name == "HassGetState":
            return ("_state_query", None)  # Special marker

        
        # On/Off toggle
        if intent_name in ("HassTurnOn", "HassTurnOff"):
            state = "on" if intent_name == "HassTurnOn" else "off"
            return ("toggle", state)
        
        # Light brightness
        if intent_name == "HassLightSet":
            direction = params.get("direction")
            brightness = params.get("brightness")
            
            if direction == "increased":
                return ("brightness_up", None)
            elif direction == "decreased":
                return ("brightness_down", None)
            elif brightness is not None:
                return ("brightness_set", brightness)
            return ("toggle", "on")  # Fallback
        
        # Cover position
        if intent_name == "HassSetPosition":
            direction = params.get("direction")
            position = params.get("position")
            
            if direction == "increased":
                return ("open", None)
            elif direction == "decreased":
                return ("close", None)
            elif position is not None:
                return ("position", position)
            return ("toggle", "on")  # Fallback
        
        # Climate temperature
        if intent_name == "HassClimateSetTemperature":
            temp = params.get("temperature")
            return ("set_temperature", temp)
        
        # Vacuum
        if intent_name == "HassVacuumStart":
            area = params.get("area")
            if area:
                return ("start_area", None)
            return ("start", None)
        
        # Temporary/Delayed control - use toggle with duration info
        if intent_name in ("TemporaryControl", "DelayedControl"):
            return ("toggle", "on")
        
        # Default fallback
        return ("toggle", "on")
