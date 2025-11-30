import logging
from typing import Any, Dict
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class ClarificationCapability(Capability):
    """Split or rephrase unclear commands."""

    name = "clarification"
    description = "Turn compound commands into atomic smart-home actions."

    PROMPT = {
    "system": """
You are a language model that obtains intents from a German user commands for smart home control.

## Input
- user_input: A German natural language command.

## Rules
1. Split the input into a list of precise **atomic commands** in German only if the target is different.
2. Each command must describe exactly one action.
3. Use natural German phrasing such as:
    - "Schalte ... an" / "Schalte ... aus"
    - "Mache ... heller" (if it is too dark)
    - "Mache ... dunkler" (if it is too bright)
    - "Fahre ... hoch/runter"
    - "Setze ... auf ..."
    - "Wie ist ...?"
4. Keep all German words exactly as spoken by the user (e.g. if they say "Dusche", keep "Dusche").
5. If an area is not explicitly mentioned, do not invent or guess one.
6. Output only a JSON array of strings, each string being a precise German instruction.

## Indirect Command Examples
Input: "Im Wohnzimmer ist es zu dunkel"
Output: ["Mache das Licht im Wohnzimmer heller"]

Input: "Es ist zu hell in der Küche"
Output: ["Mache das Licht in der Küche dunkler"]

## Multi-Command Examples
Input: "Mach das Licht im Wohnzimmer an und die Jalousien runter"
Output: ["Schalte das Licht im Wohnzimmer an", "Fahre die Jalousien im Wohnzimmer runter"]

Input: "Öffne den Rolladen im Büro zu 5%"
Output: ["Öffne den Rolladen im Büro zu 5%"]
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        _LOGGER.debug("[Clarification] Splitting or refining: %s", user_input.text)
        return await self._safe_prompt(self.PROMPT, {"user_input": user_input.text})
