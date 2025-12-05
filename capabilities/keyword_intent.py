import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from custom_components.multistage_assist.conversation_utils import LIGHT_KEYWORDS, COVER_KEYWORDS, SENSOR_KEYWORDS, CLIMATE_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """Derive intent/domain from keywords."""
    name = "keyword_intent"

    DOMAIN_KEYWORDS = {
        "light": list(LIGHT_KEYWORDS.values()) + list(LIGHT_KEYWORDS.keys()),
        "cover": list(COVER_KEYWORDS.values()) + list(COVER_KEYWORDS.keys()),
        "sensor": list(SENSOR_KEYWORDS.values()) + list(SENSOR_KEYWORDS.keys()) + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.values()) + list(CLIMATE_KEYWORDS.keys()),
    }

    INTENT_DATA = {
        "light": {
            "intents": ["HassTurnOn", "HassTurnOff", "HassLightSet", "HassGetState"],
            "rules": "brightness: 'step_up'/'step_down' if no number. 0-100 otherwise."
        },
        "cover": {"intents": ["HassTurnOn", "HassTurnOff", "HassSetPosition", "HassGetState"]},
        "sensor": {"intents": ["HassGetState"]},
        "climate": {"intents": ["HassClimateSetTemperature", "HassTurnOn", "HassTurnOff", "HassGetState"]},
    }

    SCHEMA = {
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

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text
        domain = self._detect_domain(text)
        if not domain: return {}

        meta = self.INTENT_DATA.get(domain)
        
        # Improved System Prompt
        system = f"""Select Home Assistant intent for domain '{domain}'.
Allowed: {', '.join(meta.get('intents', []))}
Slots: area, name, domain, floor, device_class (for sensors).
Rules: {meta.get('rules', '')}

IMPORTANT:
- Only fill 'name' if a SPECIFIC device is named (e.g. "Deckenlampe", "Spots", "Stehlampe").
- If the user says generic words like "Licht", "Lampe", "Gerät", leave 'name' EMPTY (null).
- Example: "Licht an" -> name: null. "Stehlampe an" -> name: "Stehlampe".

Return JSON: {{"intent": "...", "slots": {{...}}}}
"""
        data = await self._safe_prompt({"system": system, "schema": self.SCHEMA}, {"user_input": text})
        
        if not isinstance(data, dict) or not data.get("intent"): return {}
        
        slots = data.get("slots", {})
        if "domain" not in slots: slots["domain"] = domain
        
        return {"domain": domain, "intent": data["intent"], "slots": slots}