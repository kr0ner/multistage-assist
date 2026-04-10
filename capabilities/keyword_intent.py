import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from ..utils.german_utils import (
    DOMAIN_DESCRIPTIONS,
    IMPLICIT_PHRASES,
    GERMAN_ARTICLES,
    GERMAN_PREPOSITIONS,
)
from ..utils.fuzzy_utils import levenshtein_distance
from ..constants.entity_keywords import (
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
    description = "Determine the target domain and intent from natural language using tiered matching: 1. Exact string matching against domain keyword dictionaries 2. Fuzzy Levenshtein distance matching for typos 3. Specialized LLM reasoning for slot extraction (area, floor, parameters). Supports complex control modes like Temporary and Delayed control."

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
- 'TemporaryControl': Use this if a DURATION is specified with the preposition for temporary action.
  - 'duration': The duration string.
  - 'command': "on" or "off".
"""

    # Rule for delayed/scheduled control
    _DELAYED_RULE = """
- 'DelayedControl': Use this if action should be DELAYED/SCHEDULED:
  - Keywords indicating future time offset or absolute time (NOT temporary duration).
  - 'delay': The delay/time string.
  - 'command': "on" or "off".
"""""

    INTENT_DATA = {
        "light": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassLightSet",
                "HassGetState",
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": """For HassLightSet:
- 'command': use 'step_up' for relative brightness increase, 'step_down' for decrease.
- 'brightness': use integer 0-100 ONLY for explicit percentages.
- Do NOT put step_up/step_down in brightness slot!
"""
            + _TEMP_RULE + _DELAYED_RULE,
        },
        "cover": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassSetPosition",
                "HassGetState",
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "switch": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "fan": {
            "intents": [
                "HassTurnOn",
                "HassTurnOff",
                "HassGetState",
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": _TEMP_RULE + _DELAYED_RULE,
        },
        "media_player": {
            "intents": ["HassTurnOn", "HassTurnOff", "HassGetState"],
            "rules": "",
        },
        "sensor": {
            "intents": ["HassGetState"],
            "rules": "- device_class: required (temperature, humidity, power, energy, battery).\n- name: EMPTY unless a specific device is named.",
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
            "intents": ["HassVacuumStart", "HassVacuumReturnToBase"],
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
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": """- 'name': The automation/device name.
- If DURATION specified for temporary action, use TemporaryControl.
- If DELAYED to a future time, use DelayedControl.
""" + _TEMP_RULE + _DELAYED_RULE,
        },
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.memory = None

    def set_memory(self, memory_cap):
        self.memory = memory_cap

    SCHEMA = {
        "type": "object",
        "properties": {
            "intent": {"type": ["string", "null"]},
            "area": {"type": ["string", "null"]},
            "floor": {"type": ["string", "null"]},
            "domain": {"type": ["string", "null"]},
            "command": {"type": ["string", "null"]},
            "duration": {"type": ["string", "null"]},
            "position": {"type": ["string", "null"]},
            "brightness": {"type": ["string", "null"]},
            "temperature": {"type": ["string", "null"]},
            "device_class": {"type": ["string", "null"]},
            "state": {"type": ["string", "null"]},
            "slots": {
                "type": "object",
                "additionalProperties": True
            }
        },
        "required": ["intent", "slots"]
    }

    def _fuzzy_match_distance(self, word: str, keyword: str, max_distance: int = 2) -> Optional[int]:
        """Return edit distance if within threshold, else None.
        
        Strict same-length matching to catch typos without false positives.
        """
        if len(word) != len(keyword):
            return None
        
        # Minimum length to apply fuzzy (avoid matching "an" to "auf")
        if len(keyword) < 5:
            return 0 if word == keyword else None
        
        dist = levenshtein_distance(word, keyword)
        return dist if dist <= max_distance else None


    async def _semantic_match(self, text: str) -> Optional[str]:
        """Direct access to semantic cache for integration tests/fallbacks."""
        from .semantic_cache import SemanticCacheCapability
        # Use existing semantic_cache capability if available or direct stage1
        from ..stage1_cache import match
        
        result, score = await match(text)
        if score > 0.85: # Threshold from previous logic
            # Handle list/str from new cache results
            if isinstance(result, list):
                return result[0] if result else None
            return result
        return None

    def _detect_domain(self, text: str) -> Optional[str]:
        t = text.lower()
        words = t.split()
        
        # First pass: exact substring match
        matches = [
            d for d, kws in self.DOMAIN_KEYWORDS.items() if any(k in t for k in kws)
        ]
        
        if len(matches) == 1:
            return matches[0]
        if "climate" in matches and "sensor" in matches:
            return "climate"
        if "switch" in matches and "sensor" in matches:
            return "sensor"
        if "calendar" in matches:
            return "calendar"
        if "timer" in matches:
            return "timer"
        if "vacuum" in matches:
            return "vacuum"
        if matches:
            return matches[0]
        
        # Second pass: fuzzy match - collect ALL matches with distances, pick BEST
        candidates = []  # (domain, keyword, distance)
        for word in words:
            for domain, keywords in self.DOMAIN_KEYWORDS.items():
                for kw in keywords:
                    dist = self._fuzzy_match_distance(word, kw)
                    if dist is not None:
                        candidates.append((domain, kw, dist, word))
        
        if candidates:
            # Sort by distance (ascending), pick smallest
            candidates.sort(key=lambda x: x[2])
            best = candidates[0]
            _LOGGER.debug(
                "[KeywordIntent] Fuzzy match: '%s' → '%s' (domain=%s, dist=%d)", 
                best[3], best[1], best[0], best[2]
            )
            return best[0]
        
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
- For HassGetState: use 'state' slot to capture the QUERIED state (on/off/open/closed)."""

        personal_info = ""
        if self.memory:
            data = await self.memory.get_all_personal_data()
            if data:
                personal_info = "Known Personal Information:\n" + "\n".join(f"- {k}: {v}" for k, v in data.items()) + "\n\n"

        system = f"""You are a smart home assistant. Identify the intent and extract slots from the user's command.
{personal_info}Allowed Intents: {', '.join(intents)}
Allowed Slots: area, name, domain, floor, duration, command, device_class, position, temperature, brightness.

Rules: {meta.get('rules', '')}
- Use 'floor' for floor/level references.
- Use 'area' for room/area/location references.
- If generic device words are used without a specific name, 'name' must be EMPTY.
- Do NOT put quantifiers like 'all' in 'area' or 'name'.
- ALWAYS use one of the "Allowed Intents" exactly as written.
{get_state_instructions}
"""
        data = await self._safe_prompt(
            {"system": system, "schema": self.SCHEMA}, {"user_input": text}
        )

        if not isinstance(data, dict):
            _LOGGER.warning("[KeywordIntent] Bad data type from prompt_executor: %s", type(data))
            return {}
        if not data.get("intent"):
            _LOGGER.warning("[KeywordIntent] No intent extracted: %s", data)
            return {}

        slots = data.get("slots") or {}
        # Merge top-level properties from schema into slots (Ollama often puts them at top level)
        known_slots = [
            "area", "floor", "domain", "command", "duration", 
            "position", "brightness", "temperature", "device_class", "state"
        ]
        for prop in known_slots:
            val = data.get(prop)
            if val is not None and val != "" and not slots.get(prop):
                slots[prop] = val
        
        # Pull extra keys from top level if slots was missing but props were there
        if not slots:
            slots = {k: v for k, v in data.items() if k not in ("intent", "slots")}

        if "domain" not in slots:
            slots["domain"] = domain
            
        # Post-processing: Remove "alle" from area/name if LLM put it there
        if slots.get("area") and str(slots["area"]).lower() in ("alle", "alles", "ganze", "gesamte", "sämtliche"):
            slots["area"] = None
        if slots.get("name") and str(slots["name"]).lower() in ("alle", "alles", "ganze", "gesamte", "sämtliche"):
            slots["name"] = None

        return {"domain": domain, "intent": data["intent"], "slots": slots}
