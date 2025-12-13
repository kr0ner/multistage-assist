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
)

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """Derive intent/domain from keywords."""

    name = "keyword_intent"

    DOMAIN_KEYWORDS = {
        "light": list(LIGHT_KEYWORDS.values()) + list(LIGHT_KEYWORDS.keys()),
        "cover": list(COVER_KEYWORDS.values()) + list(COVER_KEYWORDS.keys()),
        "switch": list(SWITCH_KEYWORDS.values()) + list(SWITCH_KEYWORDS.keys()),
        "fan": list(FAN_KEYWORDS.values()) + list(FAN_KEYWORDS.keys()),
        "media_player": list(MEDIA_KEYWORDS.values()) + list(MEDIA_KEYWORDS.keys()),
        "sensor": list(SENSOR_KEYWORDS.values())
        + list(SENSOR_KEYWORDS.keys())
        + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.values()) + list(CLIMATE_KEYWORDS.keys()),
        "timer": TIMER_KEYWORDS,
        "vacuum": VACUUM_KEYWORDS,
        "calendar": CALENDAR_KEYWORDS,
    }

    # Common rule for temp control
    _TEMP_RULE = """
- 'HassTemporaryControl': Use this if a DURATION is specified (e.g. "für 10 Minuten").
  - 'duration': The duration string (e.g. "10 Minuten").
  - 'command': "on" (an/ein/auf) or "off" (aus/zu).
"""

    INTENT_DATA = {
        "light": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassLightSet",
                "HassGetState",
                "HassTemporaryControl",
            ],
            "rules": "brightness: 'step_up'/'step_down' if no number. 0-100 otherwise."
            + _TEMP_RULE,
        },
        "cover": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassSetPosition",
                "HassGetState",
                "HassTemporaryControl",
            ],
            "rules": _TEMP_RULE,
        },
        "switch": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "HassTemporaryControl",
            ],
            "rules": _TEMP_RULE,
        },
        "fan": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "HassTemporaryControl",
            ],
            "rules": _TEMP_RULE,
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
            "rules": "- duration: seconds or text.\n- name: target device.",
        },
        "vacuum": {
            "intents": ["HassVacuumStart"],
            "rules": """CRITICAL - ALWAYS fill 'mode' slot:
- mode='vacuum': saugen, staubsaugen, sauge, staubsauge, absaugen (DEFAULT)
- mode='mop': wischen, wische, nass, feucht, moppen, feuchtwischen
EXAMPLES:
  "sauge den Keller" → mode='vacuum', area='Keller'
  "wische die Küche" → mode='mop', area='Küche'
  "wische den Keller" → mode='mop', area='Keller'
- 'area': Extract ONLY the room name (without articles like 'den', 'die', 'das', 'im').
- 'floor': Floor name if specified.
- 'scope': 'GLOBAL' if whole house.""",
        },
        "calendar": {
            "intents": ["HassCalendarCreate", "HassCreateEvent"],
            "rules": "- 'summary': Event title/name.\n- 'date': Date of event.\n- 'time': Time of event.\n- 'location': Event location.\n- 'calendar': Calendar name.",
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

        system = f"""Select Home Assistant intent for domain '{domain}'.
Allowed: {', '.join(meta.get('intents', []))}
Slots: area, name, domain, floor, device_class, duration, command, mode, scope, state.
Rules: {meta.get('rules', '')}

IMPORTANT:
- Only fill 'name' if a SPECIFIC device is named.
- If generic words (Licht, Lampe), leave 'name' EMPTY.
- For HassGetState: use 'state' slot for queries like "which lights are ON" → {{"state": "on"}}
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
