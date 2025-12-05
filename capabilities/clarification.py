import logging
from typing import Any, Dict
from .base import Capability

_LOGGER = logging.getLogger(__name__)

class ClarificationCapability(Capability):
    """Split or rephrase unclear commands."""
    name = "clarification"

    PROMPT = {
    "system": """You are a smart home intent parser.
Task: Split the input into precise atomic German commands.
Rules:
1. One action per string.
2. "Zu dunkel" -> "Mache Licht heller". "Zu hell" -> "Mache Licht dunkler".
3. Use specific device names if given.

Examples:
"Licht im Bad an und Rollo runter" -> ["Schalte Licht im Bad an", "Fahre Rollo runter"]
"Im Büro ist es zu dunkel" -> ["Mache das Licht im Büro heller"]
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        _LOGGER.debug("[Clarification] Processing: %s", user_input.text)
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})