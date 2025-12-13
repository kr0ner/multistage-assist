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

CRITICAL RULES:
1. **ALWAYS** split compound commands with "und" into separate array elements.
2. Each action must be its own string in the output array.
3. "Zu dunkel" -> "Mache Licht heller". "Zu hell" -> "Mache Licht dunkler".
4. Use specific device names if given.
5. **PRESERVE** time/duration constraints (e.g., "für 5 Minuten", "für 1 Stunde").
6. **SPLIT** opposite actions (an/aus, auf/zu, heller/dunkler) in different locations into separate commands.

Output Format:
- JSON array of strings
- ONE command per string
- If input has "und", output MUST be array with 2+ items

Examples:
Input: "Licht im Bad an und Rollo runter"
Output: ["Schalte Licht im Bad an", "Fahre Rollo runter"]

Input: "Im Büro ist es zu dunkel"
Output: ["Mache das Licht im Büro heller"]

Input: "Schalte das Licht für 10 Minuten an"
Output: ["Schalte das Licht für 10 Minuten an"]

Input: "Mache das Licht in der Küche an und im Flur aus"
Output: ["Mache das Licht in der Küche an", "Mache das Licht im Flur aus"]

Input: "Stelle die Heizung im Wohnzimmer auf 22 Grad und im Schlafzimmer auf 18 Grad"
Output: ["Stelle die Heizung im Wohnzimmer auf 22 Grad", "Stelle die Heizung im Schlafzimmer auf 18 Grad"]
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, **kwargs):
        """
        Detect vague/ambiguous language and rewrite to something clearer.
        """
        text = user_input.text.strip()
        _LOGGER.debug("[Clarification] Input: %s", text)

        # Early bypass optimization: Skip LLM for very simple, short commands
        # Only applies to commands with no separators and very few words
        separators = [",", " and ", " und ", "oder", " or ", " dann "]
        text_lower = f" {text.lower()} "  # Add spaces for word boundary matching
        has_separator = any(sep in text_lower for sep in separators)

        # For very short commands without separators, bypass LLM
        # Conservative threshold to avoid breaking complex rephrasing
        word_count = len(text.split())
        is_very_simple = word_count <= 5 and not has_separator

        if is_very_simple:
            _LOGGER.debug(
                "[Clarification] Very simple command (%d words, no separators), bypassing LLM",
                word_count,
            )
            # Return same format as _safe_prompt returns: just the list
            return [text]

        _LOGGER.debug(
            "[Clarification] Calling LLM - words: %d, has_separator: %s",
            word_count,
            has_separator,
        )
        return await self._safe_prompt(
            self.PROMPT, {"user_input": text}, temperature=0.3
        )
