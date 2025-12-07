import logging
from typing import Any, Dict
from .base import Capability
from custom_components.multistage_assist.conversation_utils import _ENTITY_PLURALS, _PLURAL_CUES, _NUM_WORDS, _NUMERIC_PATTERN

_LOGGER = logging.getLogger(__name__)

class PluralDetectionCapability(Capability):
    """Detect plural references in German smart-home commands."""
    name = "plural_detection"
    
    PROMPT = {
        "system": """Detect plural in German commands.
Plural nouns or 'alle' -> true. Singular -> false.
JSON: {"multiple_entities": boolean}""",
        "schema": {"properties": {"multiple_entities": {"type": "boolean"}}}
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        text = user_input.text.lower().strip()
        
        # Fast Path
        if any(cue in text for cue in _PLURAL_CUES): return {"multiple_entities": True}
        if any(num in text for num in _NUM_WORDS) or _NUMERIC_PATTERN.search(text): return {"multiple_entities": True}

        for sing, plural in _ENTITY_PLURALS.items():
            if plural in text: return {"multiple_entities": True}
            if sing in text: return {"multiple_entities": False}

        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})