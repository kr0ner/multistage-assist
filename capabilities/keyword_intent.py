import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from custom_components.multistage_assist.conversation_utils import LIGHT_KEYWORDS, COVER_KEYWORDS, SENSOR_KEYWORDS, CLIMATE_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """
    Detect a domain from German keywords and let the LLM pick
    a specific Home Assistant intent + slots within that domain.
    """

    name = "keyword_intent"
    description = "Derive a Home Assistant intent from a single German command using keyword domains."

    # Domain → list of keywords (singular + plural) reusing plural_detection helpers.
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "light": list(LIGHT_KEYWORDS.values()) + list(LIGHT_KEYWORDS.keys()),
        "cover": list(COVER_KEYWORDS.keys()) + list(COVER_KEYWORDS.values()),
        "sensor": list(SENSOR_KEYWORDS.values()) + list(SENSOR_KEYWORDS.keys()) + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.keys()) + list(CLIMATE_KEYWORDS.values()) + ["klima"],
    }

    # Intents per domain + extra description + examples to tune the prompt.
    INTENT_DOMAINS: Dict[str, Dict[str, Any]] = {
        "light": {
            "intents": ["HassTurnOn", "HassTurnOff", "HassLightSet", "HassGetState"],
            "rules": """
- 'brightness': Integer 0-100 OR relative command string.
  - If user gives a specific number -> use the integer (e.g., 50).
  - If user says "dimmen", "dunkler", "weniger hell" without a number -> use string "step_down"
  - If user says "heller", "aufhellen", "mehr licht" without a number -> use string "step_up"
            """
        },
        "cover": {
            "intents": ["HassTurnOn", "HassTurnOff", "HassSetPosition", "HassGetState"],
        },
        "sensor": {
            "intents": ["HassGetState"],
            "rules": """
- 'device_class': REQUIRED if the user asks for a specific measurement.
  - "Temperatur", "warm", "kalt" -> device_class: "temperature"
  - "Feuchtigkeit", "Luftfeuchte" -> device_class: "humidity"
  - "Leistung", "Watt", "Verbrauch" -> device_class: "power"
  - "Energie", "kWh" -> device_class: "energy"
  - "Batterie", "Ladung" -> device_class: "battery"
- 'name': Leave EMPTY if 'device_class' is set (unless a specific device name like 'Deckenmonitor' is given).
            """,
            "examples": [
                'User: "Wie ist die Temperatur im Büro?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Büro","domain":"sensor","device_class":"temperature"}}',
                'User: "Wieviel Watt verbraucht die Küche?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Küche","domain":"sensor","device_class":"power"}}',
            ],
        },
        "climate": {
            "intents": ["HassClimateSetTemperature", "HassTurnOn", "HassTurnOff", "HassGetState"],
        },
    }

    SCHEMA: Dict[str, Any] = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        },
    }

    def _detect_domain(self, text: str) -> Optional[str]:
        t = text.lower()
        matches = [d for d, kws in self.DOMAIN_KEYWORDS.items() if any(k in t for k in kws)]
        if len(matches) == 1: return matches[0]
        if "climate" in matches and "sensor" in matches: return "climate"
        return None

    def _build_system_prompt(self, domain: str, meta: Dict[str, Any]) -> str:
        desc = meta.get("description") or ""
        intents = meta.get("intents") or []
        examples = meta.get("examples") or []
        slot_rules = meta.get("rules") or ""

        lines = [
            f"Select Home Assistant intent for domain '{domain}'.",
            f"Allowed: {', '.join(intents)}",
            "Slots: area, name, domain, floor, device_class (for sensors).",
            f"Rules: {slot_rules}",
            "",
            "IMPORTANT:",
            "- Only fill 'name' if a SPECIFIC device is named (e.g. 'Deckenlampe', 'Spots').",
            "- If the user says generic words like 'Licht', 'Lampe', 'Gerät', 'Sensor', leave 'name' EMPTY (null).",
            "",
            f"Return JSON: {{\"intent\": \"...\", \"slots\": {{...}}}}"
        ]

        if examples:
            lines.append("\nExamples:")
            for ex in examples:
                lines.append(ex)

        return "\n".join(lines)

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text
        domain = self._detect_domain(text)
        if not domain: return {}

        meta = self.INTENT_DOMAINS.get(domain) or {}
        system = self._build_system_prompt(domain, meta)
        
        data = await self._safe_prompt({"system": system, "schema": self.SCHEMA}, {"user_input": text})
        
        if not isinstance(data, dict) or not data.get("intent"): return {}
        
        slots = data.get("slots") or {}
        if "domain" not in slots: slots["domain"] = domain
        
        return {"domain": domain, "intent": data["intent"], "slots": slots}