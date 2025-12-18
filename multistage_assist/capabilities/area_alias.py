import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import area_registry as ar, floor_registry as fr
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class AreaAliasCapability(Capability):
    """
    LLM-based mapping from a user-provided location string to a HA Area OR Floor.
    """

    name = "area_alias"
    description = "Map a location string to a Home Assistant area/floor or detect global scope."

    PROMPT = {
        "system": """
You are a smart home helper that maps a user's spoken location to the correct internal Home Assistant name.

## Input
- user_query: The name spoken by the user (e.g. "Bad", "Keller", "Oben").
- candidates: A list of available names (Areas or Floors).

## Task
1. Find the candidate that best matches `user_query`.
2. Handle synonyms: "Bad" -> "Badezimmer", "Keller" -> "Untergeschoss", "Unten" -> "Erdgeschoss".
3. **Global Scope:** If the user says "Haus", "Wohnung", "Überall", "Alles", return "GLOBAL".
4. If no candidate matches plausibly, return null.

## Output (STRICT)
Return a JSON object: { "match": <string_candidate_name_or_GLOBAL_or_null> }
""",
        "schema": {
            "type": "object",
            "properties": {
                "match": {"type": ["string", "null"]},
            },
            "required": ["match"],
        },
    }

    async def run(
        self, 
        user_input, 
        search_text: str = None, 
        mode: str = "area", # "area" or "floor"
        **_: Any
    ) -> Dict[str, Any]:
        
        text = (search_text or user_input.text or "").strip()
        if not text:
            return {"area": None} # Legacy key return for compatibility

        # Check for obvious global keywords locally
        if text.lower() in ("haus", "wohnung", "daheim", "zuhause", "überall", "alles", "ganze haus"):
            return {"area": "GLOBAL", "match": "GLOBAL"}

        candidates = []
        
        if mode == "floor":
            # Load Floors
            floor_reg = fr.async_get(self.hass)
            candidates = [f.name for f in floor_reg.async_list_floors() if f.name]
        else:
            # Load Areas (Default)
            area_reg = ar.async_get(self.hass)
            candidates = [a.name for a in area_reg.async_list_areas() if a.name]

        if not candidates:
            return {"area": None, "match": None}

        # Exact match check
        text_lower = text.lower()
        for c in candidates:
            if c.lower() == text_lower:
                # Return standard keys based on mode
                return {"area": c, "match": c}

        payload = {
            "user_query": text,
            "candidates": candidates,
        }

        data = await self._safe_prompt(self.PROMPT, payload)

        if not isinstance(data, dict):
            return {"area": None, "match": None}

        matched = data.get("match")
        
        if matched == "GLOBAL":
             return {"area": "GLOBAL", "match": "GLOBAL"}

        if matched and matched in candidates:
            _LOGGER.debug("[AreaAlias] Mapped '%s' → '%s' (mode=%s)", text, matched, mode)
            # Return both keys for compat
            return {"area": matched, "match": matched}

        return {"area": None, "match": None}