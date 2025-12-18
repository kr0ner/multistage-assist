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
3. **IMPLICIT BRIGHTNESS/TEMPERATURE RULES** (VERY IMPORTANT):
   - "Zu dunkel" / "es ist dunkel" → "Mache Licht heller" (increase brightness)
   - "Zu hell" / "es ist hell" → "Mache Licht dunkler" (decrease brightness)
   - "Zu kalt" → "Mache Heizung wärmer" / "Stelle Heizung höher"
   - "Zu warm" → "Mache Heizung kälter" / "Stelle Heizung niedriger"
4. Use specific device names if given.
5. **PRESERVE** time/duration constraints (e.g., "für 5 Minuten", "für 1 Stunde").
6. **SPLIT** opposite actions (an/aus, auf/zu, heller/dunkler) in different locations into separate commands.
7. **MULTI-AREA COMMANDS**: If "und" connects AREAS/FLOORS with the same action, split into separate commands for each area.
8. **NEVER INVENT** durations or constraints that are not in the original input!

Examples:
Input: "Licht im Bad an und Rollo runter"
Output: ["Schalte Licht im Bad an", "Fahre Rollo runter"]

Input: "Im Büro ist es zu dunkel"
Output: ["Mache das Licht im Büro heller"]

Input: "Zu hell hier"
Output: ["Mache das Licht dunkler"]

Input: "Es ist zu hell im Wohnzimmer"
Output: ["Mache das Licht im Wohnzimmer dunkler"]

Input: "Schalte das Licht für 10 Minuten an"
Output: ["Schalte das Licht für 10 Minuten an"]

Input: "Mache das Licht in der Küche an und im Flur aus"
Output: ["Mache das Licht in der Küche an", "Mache das Licht im Flur aus"]

Input: "Stelle die Heizung im Wohnzimmer auf 22 Grad und im Schlafzimmer auf 18 Grad"
Output: ["Stelle die Heizung im Wohnzimmer auf 22 Grad", "Stelle die Heizung im Schlafzimmer auf 18 Grad"]

Input: "Fahre alle Rolläden im Untergeschoss und Obergeschoss herunter"
Output: ["Fahre alle Rolläden im Untergeschoss herunter", "Fahre alle Rolläden im Obergeschoss herunter"]

Input: "Alle Lichter im Erdgeschoss und ersten Stock aus"
Output: ["Schalte alle Lichter im Erdgeschoss aus", "Schalte alle Lichter im ersten Stock aus"]
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

        # Implicit phrases that ALWAYS need LLM transformation
        implicit_phrases = ["zu dunkel", "zu hell", "zu kalt", "zu warm", "zu laut", "zu leise"]
        needs_rephrasing = any(phrase in text_lower for phrase in implicit_phrases)
        
        # Calendar and timer commands should NEVER be split (unless they have compound separators)
        # The LLM confuses time ranges like "15 Uhr bis 18 Uhr" as two separate events
        calendar_keywords = ["termin", "kalender", "event", "eintrag"]
        timer_keywords = ["timer", "wecker", "erinnerung"]
        is_calendar_or_timer = any(kw in text_lower for kw in calendar_keywords + timer_keywords)
        
        # Only bypass for calendar/timer if there's NO compound separator (und/and)
        # "Timer für 10 Minuten und Licht aus" should still be split
        if is_calendar_or_timer and not has_separator:
            _LOGGER.debug("[Clarification] Calendar/timer command without separator, bypassing LLM")
            return [text]

        # For short commands without separators AND no implicit phrases, bypass LLM
        # Conservative threshold - single commands rarely need rephrasing
        word_count = len(text.split())
        is_very_simple = word_count <= 8 and not has_separator and not needs_rephrasing

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
