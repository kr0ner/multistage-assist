import logging
from typing import Any, Dict
from .base import Capability
from custom_components.multistage_assist.conversation_utils import (
    _ENTITY_PLURALS,
    _PLURAL_CUES,
    _NUM_WORDS,
    _NUMERIC_PATTERN,
)

_LOGGER = logging.getLogger(__name__)


# Pre-compute noun-only mappings from _ENTITY_PLURALS
# Converts {"das licht": "die lichter"} → {"licht": "lichter"}
_SINGULAR_NOUNS = set()
_PLURAL_NOUNS = set()
for sing, plur in _ENTITY_PLURALS.items():
    # Extract nouns from "article noun" format
    sing_noun = sing.split()[-1].lower()
    plur_noun = plur.split()[-1].lower()
    _SINGULAR_NOUNS.add(sing_noun)
    _PLURAL_NOUNS.add(plur_noun)


class PluralDetectionCapability(Capability):
    """Detect plural references in German smart-home commands."""

    name = "plural_detection"

    PROMPT = {
        "system": """You act as a detector specialized in recognizing plural references in German commands.

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
"Fahre die Rolläden herunter" => { "multiple_entities": true }
"Schalte die Lichter im Badezimmer an" => { "multiple_entities": true }
""",
        "schema": {"properties": {"multiple_entities": {"type": "boolean"}}},
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text.lower().strip()
        words = set(text.split())

        # Fast Path: Check for explicit plural cues ("alle", "beide", etc.)
        if any(cue in text for cue in _PLURAL_CUES):
            return {"multiple_entities": True}
        if any(num in text for num in _NUM_WORDS) or _NUMERIC_PATTERN.search(text):
            return {"multiple_entities": True}

        # Check for plural nouns (without articles)
        if words & _PLURAL_NOUNS:
            _LOGGER.debug("[PluralDetection] Found plural noun in: %s", words & _PLURAL_NOUNS)
            return {"multiple_entities": True}
        
        # Check for singular nouns (without articles)
        if words & _SINGULAR_NOUNS:
            _LOGGER.debug("[PluralDetection] Found singular noun in: %s", words & _SINGULAR_NOUNS)
            return {"multiple_entities": False}

        # Fallback to LLM if no fast path matches
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})
