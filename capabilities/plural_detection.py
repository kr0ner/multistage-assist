import logging
from typing import Any, Dict
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class PluralDetectionCapability(Capability):
    """Detect whether the command refers to multiple entities."""

    name = "plural_detection"
    description = "Detect plural or collective references like 'alle Lampen'."

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
        _LOGGER.debug("[PluralDetection] Checking plurality: %s", user_input.text)
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})
