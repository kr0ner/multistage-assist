import logging
import random
from typing import Any, Dict, List

from .base import Capability

_LOGGER = logging.getLogger(__name__)

class ResponseGeneratorCapability(Capability):
    """
    Generate varied, context-aware responses for completed actions.
    """

    name = "response_generator"
    description = "Generates a natural language confirmation based on the intent and entities."

    # -------------------------------------------------------------------------
    # Response Templates
    # {name} will be replaced by the entity name (e.g. "Licht im Büro")
    # -------------------------------------------------------------------------
    
    RESPONSES_ON = [
        "Alles klar, {name} ist an.",
        "Erledigt, {name} eingeschaltet.",
        "Kein Problem, {name} läuft.",
        "Geht klar, {name} ist jetzt an.",
        "Okay, {name} ist aktiviert.",
        "Ich habe {name} für dich angemacht.",
        "Befehl ausgeführt: {name} ist an.",
        "Gerne, {name} ist eingeschaltet.",
        "Sofort erledigt: {name} an.",
        "Passt, {name} ist jetzt aktiv.",
        "Erledigt.",
        "Verstanden.",
        "Wird gemacht.",
    ]

    RESPONSES_OFF = [
        "Alles klar, {name} ist aus.",
        "Erledigt, {name} ausgeschaltet.",
        "Kein Problem, {name} ist aus.",
        "Geht klar, {name} ist jetzt aus.",
        "Okay, {name} ist deaktiviert.",
        "Ich habe {name} für dich ausgemacht.",
        "Befehl ausgeführt: {name} ist aus.",
        "Gerne, {name} ist ausgeschaltet.",
        "Sofort erledigt: {name} aus.",
        "Ruhe sanft, {name} ist aus.",
        "Das war's für {name}.",
        "Strom sparen angesagt: {name} ist aus.",
    ]

    RESPONSES_SET = [
        "Erledigt, {name} ist eingestellt.",
        "Habe {name} angepasst.",
        "Okay, {name} ist auf dem gewünschten Wert.",
        "Alles klar, Einstellung für {name} übernommen.",
        "Geht klar, {name} ist gesetzt.",
        "Gemacht.",
        "Die Einstellung für {name} ist aktiv.",
        "Passt so.",
        "Wunsch erfüllt: {name} eingestellt.",
        "Gern geschehen.",
    ]

    # Fallback for unknown intents
    RESPONSES_GENERIC = [
        "Erledigt.",
        "Alles klar.",
        "Kein Problem.",
        "Befehl ausgeführt.",
        "Das habe ich gemacht.",
        "Gerne.",
        "Okay.",
        "Wird erledigt.",
        "Verstanden.",
        "Check.",
    ]

    def _get_template_pool(self, intent_name: str) -> List[str]:
        if intent_name == "HassTurnOn":
            return self.RESPONSES_ON
        if intent_name == "HassTurnOff":
            return self.RESPONSES_OFF
        if intent_name in ("HassLightSet", "HassSetPosition", "HassClimateSetTemperature"):
            return self.RESPONSES_SET
        return self.RESPONSES_GENERIC

    async def run(
        self, 
        user_input, 
        intent_name: str, 
        entity_ids: List[str], 
        **_: Any
    ) -> Dict[str, Any]:
        
        # Resolve friendly names
        names = []
        for eid in entity_ids:
            st = self.hass.states.get(eid)
            if st:
                names.append(st.attributes.get("friendly_name") or eid)
            else:
                names.append(eid)
        
        # Join names naturally (e.g. "Licht A und Licht B")
        if len(names) > 1:
            name_str = f"{', '.join(names[:-1])} und {names[-1]}"
        elif names:
            name_str = names[0]
        else:
            name_str = "das Gerät"

        # Pick a random template
        pool = self._get_template_pool(intent_name)
        template = random.choice(pool)

        # Fill slots
        response_text = template.replace("{name}", name_str)

        _LOGGER.debug("[ResponseGenerator] Generated: '%s'", response_text)
        return {"message": response_text}
