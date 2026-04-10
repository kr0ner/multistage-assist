import logging
from typing import Any, Dict
from .base import Capability
from ..conversation_utils import (
    _ENTITY_PLURALS,
    _PLURAL_CUES,
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
    description = "Detect if a command refers to multiple entities (plural) or a single specific device. Employs: 1. Keyword check ('alle', 'beide') 2. Plural noun dictionary lookup 3. Singular noun dictionary lookup 4. LLM reasoning as final fallback. Prevents unnecessary disambiguation when the user intend to target a group."

    PROMPT = {
        "system": """Detect whether a German smart home command targets multiple entities (plural) or a single one.

Rules:
- Plural nouns or quantifiers ("alle", "beide", "sämtliche") → true
- Singular nouns (with "das", "der", "die" + singular) → false
- If uncertain → return empty JSON
""",
        "schema": {"properties": {"multiple_entities": {"type": "boolean"}}},
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text.lower().strip()
        words = set(text.split())

        # Fast Path: Check for explicit plural cues ("alle", "beide", etc.)
        if any(cue in text for cue in _PLURAL_CUES):
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
