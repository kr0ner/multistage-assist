import logging
from typing import Any, Dict, Optional, List
from .base import Capability
from custom_components.multistage_assist.conversation_utils import (
    LIGHT_KEYWORDS, COVER_KEYWORDS, SENSOR_KEYWORDS, CLIMATE_KEYWORDS,
    SWITCH_KEYWORDS, FAN_KEYWORDS, MEDIA_KEYWORDS, TIMER_KEYWORDS
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
        "sensor": list(SENSOR_KEYWORDS.values()) + list(SENSOR_KEYWORDS.keys()) + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.values()) + list(CLIMATE_KEYWORDS.keys()),
        "timer": TIMER_KEYWORDS,
    }

    INTENT_DATA = {
        "light": {"intents": ["HassTurnOn", "HassTurnOff", "HassLightSet", "HassGetState"], "rules": "brightness: 'step_up'/'step_down' if no number."},
        "cover": {"intents": ["HassTurnOn", "HassTurnOff", "HassSetPosition", "HassGetState"], "rules": ""},
        "switch": {"intents": ["HassTurnOn", "HassTurnOff", "HassGetState"], "rules": ""},
        "fan": {"intents": ["HassTurnOn", "HassTurnOff", "HassGetState"], "rules": ""},
        "media_player": {"intents": ["HassTurnOn", "HassTurnOff", "HassGetState"], "rules": ""},
        "sensor": {"intents": ["HassGetState"], "rules": "- device_class: required (temperature, humidity, power, energy, battery).\n- name: EMPTY unless specific."},
        "climate": {"intents": ["HassClimateSetTemperature", "HassTurnOn", "HassTurnOff", "HassGetState"], "rules": ""},
        "timer": {"intents": ["HassTimerSet"], "rules": "- duration: seconds or text.\n- name: target device."},
    }

    SCHEMA = {"properties": {"intent": {"type": ["string", "null"]}, "slots": {"type": "object"}}}

    def _detect_domain(self, text: str) -> Optional[str]:
        t = text.lower()
        matches = [d for d, kws in self.DOMAIN_KEYWORDS.items() if any(k in t for k in kws)]
        if len(matches) == 1: return matches[0]
        if "climate" in matches and "sensor" in matches: return "climate"
        if "timer" in matches: return "timer"
        return matches[0] if matches else None

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text
        domain = self._detect_domain(text)
        if not domain: return {}

        meta = self.INTENT_DATA.get(domain) or {}
        system = f"""Select Home Assistant intent for domain '{domain}'.
Allowed: {', '.join(meta.get('intents', []))}
Slots: area, name, domain, floor, device_class, duration.
Rules: {meta.get('rules', '')}
IMPORTANT: Only fill 'name' if specific device named. For generic terms, leave 'name' EMPTY.
Return JSON: {{"intent": "...", "slots": {{...}}}}"""
        
        data = await self._safe_prompt({"system": system, "schema": self.SCHEMA}, {"user_input": text})
        if not isinstance(data, dict) or not data.get("intent"): return {}
        
        slots = data.get("slots") or {}
        if "domain" not in slots: slots["domain"] = domain
        return {"domain": domain, "intent": data["intent"], "slots": slots}