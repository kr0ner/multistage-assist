import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import area_registry as ar
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class AreaAliasCapability(Capability):
    """
    LLM-based mapping from a user-provided location string (e.g. "Bad") 
    to one of the existing Home Assistant areas.
    """

    name = "area_alias"
    description = "Map a German location string to a Home Assistant area."

    PROMPT = {
        "system": """
You are a smart home helper that maps a user's spoken room name to the correct internal Home Assistant area name.

## Input
- user_query: The name of the room as spoken by the user (e.g. "Bad", "Küche", "Gästeklo").
- areas: A list of available area names in the system.

## Task
1. Find the area in `areas` that best matches `user_query`.
2. Handle synonyms, abbreviations, and partial matches (e.g., "Bad" -> "Badezimmer", "Unten" -> "Flur Unten").
3. If multiple areas match (e.g., "Bad" matches "Gäste Bad" and "Badezimmer"), prefer the main/general one unless the query is specific.
4. If no area matches plausibly, return null.

## Output (STRICT)
Return a JSON object: { "area": <string_from_list_or_null> }
""",
        "schema": {
            "type": "object",
            "properties": {
                "area": {"type": ["string", "null"]},
            },
            "required": ["area"],
        },
    }

    async def run(self, user_input, search_text: str = None, **_: Any) -> Dict[str, Any]:
        """
        Run the area alias LLM mapping.
        If search_text is provided, we map THAT. Otherwise we map user_input.text.
        """
        # If specific text to resolve is passed (e.g. "Bad"), use it. 
        # Otherwise fallback to full utterance (less precise).
        text = (search_text or user_input.text or "").strip()
        
        if not text:
            return {"area": None}

        # Load HA areas
        area_reg = ar.async_get(self.hass)
        areas: List[str] = [a.name for a in area_reg.async_list_areas() if a.name]

        if not areas:
            _LOGGER.debug("[AreaAlias] No areas defined in HA → cannot map.")
            return {"area": None}

        # Quick check: exact match?
        text_lower = text.lower()
        for a in areas:
            if a.lower() == text_lower:
                return {"area": a}

        payload = {
            "user_query": text,
            "areas": areas,
        }

        _LOGGER.debug("[AreaAlias] Mapping query=%r to one of areas=%s", text, areas)
        data = await self._safe_prompt(self.PROMPT, payload)

        if not isinstance(data, dict):
            return {"area": None}

        mapped = data.get("area")
        if mapped and mapped in areas:
            _LOGGER.debug("[AreaAlias] Mapped '%s' → '%s'", text, mapped)
            return {"area": mapped}

        _LOGGER.debug("[AreaAlias] No valid mapping found for '%s'", text)
        return {"area": None}