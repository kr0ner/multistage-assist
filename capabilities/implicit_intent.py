import logging
from typing import List

from .base import Capability
from ..constants.messages_de import IMPLICIT_PHRASES, IMPLICIT_INTENT_MAPPINGS

_LOGGER = logging.getLogger(__name__)


# Direct mappings for fast path (skip LLM)
# Mappings for common implicit phrases to explicit commands
DIRECT_MAPPINGS = IMPLICIT_INTENT_MAPPINGS


class ImplicitIntentCapability(Capability):
    """Detect and rephrase implicit intents (e.g., 'too dark').
    
    Handles vague commands where the user states a problem rather than an action.
    Example: "Es ist zu dunkel" -> "Mach das Licht heller"
    """

    name = "implicit_intent"
    description = "Handles implicit intents like 'too dark' or 'too cold' by translating them into explicit smart home commands. Uses fast path for common phrases and LLM for complex context-aware rephrasing."

    PROMPT = {
        "system": """You are a smart home intent parser.
Task: Translate implicit statements into explicit German commands.

CRITICAL RULES:
1. **IMPLICIT BRIGHTNESS/TEMPERATURE RULES** (VERY IMPORTANT):
   - "Zu dunkel" / "es ist dunkel" → "Mache Licht heller" (increase brightness)
   - "Zu hell" / "es ist hell" → "Mache Licht dunkler" (decrease brightness)
   - "Zu kalt" → "Mache Heizung wärmer" / "Stelle Heizung höher"
   - "Zu warm" → "Mache Heizung kälter" / "Stelle Heizung niedriger"
2. Use specific device names if given.
3. **NEVER INVENT** durations or constraints that are not in the original input!

Examples:
Input: "Im Büro ist es zu dunkel"
Output: ["Mache das Licht im Büro heller"]

Input: "Es ist zu hell im Wohnzimmer"
Output: ["Mache das Licht im Wohnzimmer dunkler"]
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, **kwargs) -> List[str]:
        """Detect and rephrase implicit commands."""
        text = user_input.text.strip()
        text_lower = text.lower().strip(".,!?")
        
        # 1. Fast Path: Check direct mappings
        # Check if the input IS one of the mappings
        if text_lower in DIRECT_MAPPINGS:
            mapped = DIRECT_MAPPINGS[text_lower]
            _LOGGER.info("[ImplicitIntent] Fast path match: '%s' -> ['%s']", text, mapped)
            return [mapped]
            
        # Check for implicit phrases
        if any(phrase in text_lower for phrase in IMPLICIT_PHRASES):
            _LOGGER.debug("[ImplicitIntent] Implicit phrase detected, calling LLM")
            return await self._safe_prompt(
                self.PROMPT, {"user_input": text}, temperature=0.3
            )
            
        return [text]
