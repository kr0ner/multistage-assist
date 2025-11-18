import re
import logging
from typing import Any, Dict, Optional
from .base import Capability

_LOGGER = logging.getLogger(__name__)

# --- Group-specific singular → plural pairs ------------------------

# Lights: used both for plural detection and keyword-based intent grouping
LIGHT_KEYWORDS: Dict[str, str] = {
    "licht": "lichter",
    "lampe": "lampen",
}

# Covers: used both for plural detection and keyword-based intent grouping
COVER_KEYWORDS: Dict[str, str] = {
    "rollladen": "rollläden",
    "rollo": "rollos",
    "jalousie": "jalousien",
}

# Other entity types (kept here for plural detection only, but can be grouped later)
OTHER_ENTITY_PLURALS: Dict[str, str] = {
    "das fenster": "die fenster",            # article-based disambiguation
    "tür": "türen",
    "tor": "tore",
    "steckdose": "steckdosen",
    "heizung": "heizungen",
    "ventilator": "ventilatoren",
    "der lautsprecher": "die lautsprecher",  # article-based disambiguation
    "gerät": "geräte",
}

# --- Common singular → plural mapping for smart home entities ---
# This is the union of all helpers above
_ENTITY_PLURALS: Dict[str, str] = {
    **LIGHT_KEYWORDS,
    **COVER_KEYWORDS,
    **OTHER_ENTITY_PLURALS,
}

# --- Quantifier or numeric plural cues ---
_PLURAL_CUES = {
    "alle", "sämtliche", "mehrere", "beide", "beiden", "viele", "verschiedene"
}

# --- Number words (for numeric plurals like "zwei Lampen") ---
_NUM_WORDS = {
    "zwei", "drei", "vier", "fünf", "sechs", "sieben",
    "acht", "neun", "zehn", "elf", "zwölf"
}

_NUMERIC_PATTERN = re.compile(r"\b\d+\b")


def _fast_plural(user_text: Optional[str]) -> Optional[Dict[str, bool]]:
    """Fast, rule-based plural detection using article-aware mappings."""
    if not user_text:
        return None

    t = user_text.lower().strip()

    # 1️⃣ Quantifier or numeric plural cues
    if any(cue in t for cue in _PLURAL_CUES):
        return {"multiple_entities": True}
    if any(num in t for num in _NUM_WORDS) or _NUMERIC_PATTERN.search(t):
        return {"multiple_entities": True}

    # 2️⃣ Direct mapping check
    for singular, plural in _ENTITY_PLURALS.items():
        if plural in t:
            return {"multiple_entities": True}
        if singular in t:
            return {"multiple_entities": False}

    # 3️⃣ Nothing matched → inconclusive
    return None


class PluralDetectionCapability(Capability):
    """Detect whether a command refers to multiple entities (mapping-based fast path)."""

    name = "plural_detection"
    description = "Detect plural references in German smart-home commands."

    PROMPT = {
        "system": """
You act as a detector specialized in recognizing plural references in German commands.

## Rule
- Plural nouns or use of *"alle"* → respond with `true`
- Singular nouns → respond with `false`
- Uncertainty → respond with empty JSON

## Examples
"Schalte die Lampen an" => { "multiple_entities": true }
"Schalte das Licht aus" => { "multiple_entities": false }
"Öffne alle Rolläden" => { "multiple_entities": true }
"Senke den Rolladen im Büro" => { "multiple_entities": false }
"Schließe alle Fenster im Obergeschoss" => { "multiple_entities": true }
""",
        "schema": {
            "properties": {
                "multiple_entities": {"type": "boolean"}
            },
        },
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        _LOGGER.debug("[PluralDetection] Checking plurality (fast-path): %s", user_input.text)

        fast = _fast_plural(user_input.text)
        if fast is not None:
            _LOGGER.debug("[PluralDetection] Fast-path result: %s", fast)
            return fast

        _LOGGER.debug("[PluralDetection] Fast-path inconclusive → LLM fallback")
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})
