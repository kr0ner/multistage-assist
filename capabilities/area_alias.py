import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import area_registry as ar
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class AreaAliasCapability(Capability):
    """
    LLM-based mapping from a user-provided location string to a HA area.
    Can also return "GLOBAL" for whole-home contexts.
    """

    name = "area_alias"
    description = "Map a location string to a Home Assistant area or detect global scope."

    PROMPT = {
        "system": """
You are a smart home helper that maps a user's spoken room name to the correct internal Home Assistant area name.

## Input
- user_query: The name of the room as spoken by the user (e.g. "Bad", "Küche", "Haus").
- areas: A list of available area names in the system.

## Task
1. Find the area in `areas` that best matches `user_query`.
2. Handle synonyms: "Bad" -> "Badezimmer", "Unten" -> "Flur Unten".
3. **Global Scope:** If the user says "Haus", "Wohnung", "Überall", "Alles" or implies the entire home, return "GLOBAL".
4. If no area matches plausibly, return null.

## Output (STRICT)
Return a JSON object: { "area": <string_area_name_or_GLOBAL_or_null> }
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
        text = (search_text or user_input.text or "").strip()
        if not text:
            return {"area": None}

        # Check for obvious global keywords locally to save an LLM call
        if text.lower() in ("haus", "wohnung", "daheim", "zuhause", "überall", "alles"):
            _LOGGER.debug("[AreaAlias] Detected global scope keyword '%s'", text)
            return {"area": "GLOBAL"}

        area_reg = ar.async_get(self.hass)
        areas: List[str] = [a.name for a in area_reg.async_list_areas() if a.name]

        if not areas:
            return {"area": None}

        # Exact match check
        text_lower = text.lower()
        for a in areas:
            if a.lower() == text_lower:
                return {"area": a}

        payload = {
            "user_query": text,
            "areas": areas,
        }

        data = await self._safe_prompt(self.PROMPT, payload)

        if not isinstance(data, dict):
            return {"area": None}

        mapped = data.get("area")
        
        if mapped == "GLOBAL":
             _LOGGER.debug("[AreaAlias] Mapped '%s' → GLOBAL scope", text)
             return {"area": "GLOBAL"}

        if mapped and mapped in areas:
            _LOGGER.debug("[AreaAlias] Mapped '%s' → '%s'", text, mapped)
            return {"area": mapped}

        return {"area": None}
