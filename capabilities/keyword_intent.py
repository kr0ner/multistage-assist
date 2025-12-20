import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from custom_components.multistage_assist.conversation_utils import (
    LIGHT_KEYWORDS,
    COVER_KEYWORDS,
    SENSOR_KEYWORDS,
    CLIMATE_KEYWORDS,
    SWITCH_KEYWORDS,
    FAN_KEYWORDS,
    MEDIA_KEYWORDS,
    TIMER_KEYWORDS,
    VACUUM_KEYWORDS,
    CALENDAR_KEYWORDS,
    AUTOMATION_KEYWORDS,
)

_LOGGER = logging.getLogger(__name__)


def _extract_nouns(keywords_dict: Dict[str, str]) -> List[str]:
    """Extract nouns from 'article noun' format keywords."""
    nouns = []
    for key, value in keywords_dict.items():
        nouns.append(key.split()[-1])
        nouns.append(value.split()[-1])
    return nouns


class KeywordIntentCapability(Capability):
    """Derive intent/domain from keywords."""

    name = "keyword_intent"

    DOMAIN_KEYWORDS = {
        "light": _extract_nouns(LIGHT_KEYWORDS),
        "cover": _extract_nouns(COVER_KEYWORDS),
        "switch": _extract_nouns(SWITCH_KEYWORDS),
        "fan": _extract_nouns(FAN_KEYWORDS),
        "media_player": _extract_nouns(MEDIA_KEYWORDS),
        "sensor": _extract_nouns(SENSOR_KEYWORDS) + ["grad", "warm", "kalt", "wieviel"],
        "climate": _extract_nouns(CLIMATE_KEYWORDS),
        "timer": TIMER_KEYWORDS,
        "vacuum": VACUUM_KEYWORDS,
        "calendar": CALENDAR_KEYWORDS,
        "automation": AUTOMATION_KEYWORDS,
    }

    # Common rule for temp control
    _TEMP_RULE = """
- 'HassTemporaryControl': Use this if a DURATION is specified with "für" (e.g. "für 10 Minuten").
  - 'duration': The duration string (e.g. "10 Minuten").
  - 'command': "on" (an/ein/auf) or "off" (aus/zu).
"""

    # Rule for delayed/scheduled control
    _DELAYED_RULE = """
- 'HassDelayedControl': Use this if action should be DELAYED/SCHEDULED:
  - Keywords: "in X Minuten", "um X Uhr" (NOT "für" - that's temporary!)
  - 'delay': The delay/time string (e.g. "10 Minuten", "15:30", "15 Uhr").
  - 'command': "on" (an/ein/auf) or "off" (aus/zu).
  - Examples:
    - "Schalte IN 10 Minuten das Licht aus" → HassDelayedControl, delay="10 Minuten"
    - "Mach UM 15 Uhr das Licht an" → HassDelayedControl, delay="15 Uhr"
"""

    INTENT_DATA = {
        "light": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassLightSet",
                "HassGetState",
                "HassTemporaryControl",
                "HassDelayedControl",
            ],
            "rules": "brightness: 'step_up'/'step_down' if no number. 0-100 otherwise."
            + _TEMP_RULE + _DELAYED_RULE,
        },
        "cover": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassSetPosition",
                "HassGetState",
                "HassTemporaryControl",
                "HassDelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "switch": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "HassTemporaryControl",
                "HassDelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "fan": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "HassTemporaryControl",
                "HassDelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "media_player": {
            "intents": ["HassTurnOn", "HassTurnOff", "HassGetState"],
            "rules": "",
        },
        "sensor": {
            "intents": ["HassGetState"],
            "rules": "- device_class: required (temperature, humidity, power, energy, battery).\n- name: EMPTY unless specific.",
        },
        "climate": {
            "intents": [
                "HassClimateSetTemperature",
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
            ],
            "rules": "",
        },
        "timer": {
            "intents": ["HassTimerSet"],
            "rules": "",
        },
        "vacuum": {
            "intents": ["HassVacuumStart"],
            "rules": "",
        },
        "calendar": {
            "intents": ["HassCalendarCreate", "HassCreateEvent"],
            "rules": "",
        },
        "automation": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassTemporaryControl",
                "HassDelayedControl",
            ],
            "rules": """- 'name': The automation/device name.
- If DURATION specified with "für", use HassTemporaryControl.
- If DELAYED with "in X Minuten" or "um X Uhr", use HassDelayedControl.
""" + _TEMP_RULE + _DELAYED_RULE,
        },
    }

    SCHEMA = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        }
    }

    def _detect_domain(self, text: str) -> Optional[str]:
        t = text.lower()
        matches = [
            d for d, kws in self.DOMAIN_KEYWORDS.items() if any(k in t for k in kws)
        ]
        if len(matches) == 1:
            return matches[0]
        if "climate" in matches and "sensor" in matches:
            return "climate"
        # Calendar before timer - calendar keywords are more specific
        if "calendar" in matches:
            return "calendar"
        if "timer" in matches:
            return "timer"
        if "vacuum" in matches:
            return "vacuum"
        if matches:
            return matches[0]
        return None

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text
        domain = self._detect_domain(text)
        if not domain:
            return {}

        meta = self.INTENT_DATA.get(domain) or {}
        intents = meta.get('intents', [])
        
        # Build conditional instructions
        get_state_instructions = ""
        if "HassGetState" in intents:
            get_state_instructions = """
- For HassGetState: use 'state' slot to capture the QUERIED state:
  - "Sind alle Lichter aus?" → {{"state": "off"}}
  - "Sind alle Lichter an?" → {{"state": "on"}}
  - "Ist das Rollo geschlossen?" → {{"state": "closed"}}
  - "Ist das Rollo offen?" → {{"state": "open"}}"""

        system = f"""Select Home Assistant intent for domain '{domain}'.
Allowed: {', '.join(intents)}
Slots (only include if non-empty): area, name, domain, floor, duration, command.
Rules: {meta.get('rules', '')}

IMPORTANT:
- Only fill 'name' if a SPECIFIC device is named (e.g., "Schreibtischlampe", "Deckenleuchte").
- If generic words (Licht, Lampe, Rollo), leave 'name' EMPTY.{get_state_instructions}
- **FLOOR vs AREA** (CRITICAL):
  - Use 'floor' for: Erdgeschoss, Obergeschoss, Untergeschoss, Keller, EG, OG, UG, oben, unten, erster/zweiter Stock
  - Use 'area' for rooms: Küche, Bad, Büro, Wohnzimmer, Schlafzimmer
  - Examples:
    - "Licht im Obergeschoss an" → {{"floor": "Obergeschoss"}} NOT area!
    - "Rollläden im OG runter" → {{"floor": "OG"}}
    - "Licht im Büro an" → {{"area": "Büro"}}
"""
        data = await self._safe_prompt(
            {"system": system, "schema": self.SCHEMA}, {"user_input": text}
        )

        if not isinstance(data, dict) or not data.get("intent"):
            return {}

        slots = data.get("slots") or {}
        if "domain" not in slots:
            slots["domain"] = domain

        return {"domain": domain, "intent": data["intent"], "slots": slots}
