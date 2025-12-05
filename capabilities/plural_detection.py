import logging
from typing import Any, Dict
from .base import Capability
from custom_components.multistage_assist.conversation_utils import _ENTITY_PLURALS

_LOGGER = logging.getLogger(__name__)

class PluralDetectionCapability(Capability):
    """Detect plural references (fast path + LLM fallback)."""
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
        if any(w in text for w in ["alle", "beide", "viele"]): return {"multiple_entities": True}
        for sing, plural in _ENTITY_PLURALS.items():
            if plural in text: return {"multiple_entities": True}
            if sing in text: return {"multiple_entities": False}

        # LLM Fallback (Minimal prompt for speed)
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})