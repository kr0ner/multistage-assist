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
- 'TemporaryControl': Use this if a DURATION is specified with "für" (e.g. "für 10 Minuten").
  - 'duration': The duration string (e.g. "10 Minuten").
  - 'command': "on" (an/ein/auf) or "off" (aus/zu).
"""

    # Rule for delayed/scheduled control
    _DELAYED_RULE = """
- 'DelayedControl': Use this if action should be DELAYED/SCHEDULED:
  - Keywords: "in X Minuten", "um X Uhr" (NOT "für" - that's temporary!)
  - 'delay': The delay/time string (e.g. "10 Minuten", "15:30", "15 Uhr").
  - 'command': "on" (an/ein/auf) or "off" (aus/zu).
  - Examples:
    - "Schalte IN 10 Minuten das Licht aus" → DelayedControl, delay="10 Minuten"
    - "Mach UM 15 Uhr das Licht an" → DelayedControl, delay="15 Uhr"
"""

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
- 'command': use 'step_up' (heller) or 'step_down' (dunkler) for relative changes.
- 'brightness': use integer 0-100 ONLY for explicit percentages (e.g., "50 Prozent").
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
                "TemporaryControl",
                "DelayedControl",
            ],
            "rules": """- 'name': The automation/device name.
- If DURATION specified with "für", use TemporaryControl.
- If DELAYED with "in X Minuten" or "um X Uhr", use DelayedControl.
""" + _TEMP_RULE + _DELAYED_RULE,
        },
    }

    SCHEMA = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        }
    }

    def _levenshtein(self, word: str, keyword: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(word) > len(keyword):
            word, keyword = keyword, word
        
        distances = range(len(word) + 1)
        for i2, c2 in enumerate(keyword):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(word):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((distances[i1], distances[i1 + 1], new_distances[-1])))
            distances = new_distances
        
        return distances[-1]

    def _fuzzy_match_distance(self, word: str, keyword: str, max_distance: int = 2) -> Optional[int]:
        """Return edit distance if within threshold, else None.
        
        Handles typos like:
        - Character swaps: "lihct" → "licht"
        - Missing chars: "rolläden" → "rollläden"
        - Extra chars: "lichtt" → "licht"
        
        Only for words of similar length to avoid false positives.
        """
        # Length check - avoid matching short words to long ones
        if abs(len(word) - len(keyword)) > max_distance:
            return None
        
        # Minimum length to apply fuzzy (avoid matching "an" to "auf")
        if len(keyword) < 5:
            return 0 if word == keyword else None
        
        dist = self._levenshtein(word, keyword)
        return dist if dist <= max_distance else None

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
