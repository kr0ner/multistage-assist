"""Capability for detecting and responding to yes/no questions about entity states."""

import logging
from typing import Any, Dict, List, Optional

from .base import Capability
from ..constants.entity_keywords import DOMAIN_NAMES

_LOGGER = logging.getLogger(__name__)


class YesNoResponseCapability(Capability):
    """Handle yes/no questions about entity states."""

    name = "yes_no_response"

    # Fast path: keyword-based detection
    YES_NO_PREFIXES = [
        "ist ",
        "sind ",
        "gibt es ",
        "hat ",
        "haben ",
        "kann ",
        "können ",
        "darf ",
        "dürfen ",
        "soll ",
        "sollen ",
        "wird ",
        "werden ",
    ]

    SCHEMA = {"properties": {"is_yes_no": {"type": "boolean"}}}

    async def run(
        self,
        user_input,
        *,
        domain: str,
        state: str,
        entity_ids: List[str],
        **_: Any,
    ) -> Optional[str]:
        """
        Generate yes/no response for state query.

        Args:
            user_input: User's original query
            domain: Entity domain (light, cover, etc.)
            state: Requested state (on, off, open, closed)
            entity_ids: List of entity IDs matching the state

        Returns:
            Response text if yes/no question, None otherwise
        """
        text = user_input.text.lower().strip()

        # Detection: fast path
        is_yes_no = self._detect_yes_no_fast(text)

        # Detection: LLM fallback for edge cases
        if not is_yes_no and "?" in text:
            _LOGGER.debug("[YesNoResponse] Using LLM fallback for: %s", text)
            is_yes_no = await self._detect_yes_no_llm(text)

        if not is_yes_no:
            return None  # Not a yes/no question

        # Generate response
        return self._generate_response(domain, state, entity_ids)

    def _detect_yes_no_fast(self, text: str) -> bool:
        """Fast keyword-based detection."""
        return any(text.startswith(prefix) for prefix in self.YES_NO_PREFIXES)

    async def _detect_yes_no_llm(self, text: str) -> bool:
        """LLM-based detection for edge cases."""
        system = """Is this a yes/no question in German?

Examples:
- "Ist ein Licht an?" → YES (is a light on?)
- "Gibt es offene Fenster?" → YES (are there open windows?)
- "Welche Lichter sind an?" → NO (which lights are on?)
- "Schalte das Licht an" → NO (turn on the light)
"""
        data = await self._safe_prompt(
            {"system": system, "schema": self.SCHEMA}, {"text": text}
        )
        return data.get("is_yes_no", False) if data else False

    def _generate_response(self, domain: str, state: str, entity_ids: List[str]) -> str:
        """Generate natural German yes/no response."""
        domain_word = DOMAIN_NAMES.get(domain, "Gerät")

        # No matches - negative response
        if not entity_ids:
            return f"Nein, kein {domain_word} ist {state}."

        # Get friendly names
        names = [
            self.hass.states.get(eid).attributes.get("friendly_name", eid)
            for eid in entity_ids
        ]

        # Correct verb conjugation
        verb = "ist" if len(names) == 1 else "sind"

        # Format list with German grammar
        name_list = self._format_list(names)

        return f"Ja, {name_list} {verb} {state}."

    def _format_list(self, names: List[str]) -> str:
        """Format list of names with natural German grammar."""
        if len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} und {names[1]}"
        else:
            # Oxford comma style: "A, B und C"
            return f"{', '.join(names[:-1])} und {names[-1]}"
