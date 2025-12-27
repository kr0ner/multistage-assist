"""Area and Floor resolver capability.

Provides area/floor name resolution with:
- Memory-based alias lookup
- Fuzzy matching on HA area/floor names  
- LLM fallback for complex cases

Renamed from area_alias.py and expanded with find_area/find_floor from entity_resolver.
"""

import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import area_registry as ar, floor_registry as fr
from .base import Capability
from ..constants.messages_de import GLOBAL_KEYWORDS
from ..constants.domain_config import FLOOR_ALIASES_DE
from ..utils.german_utils import canonicalize

_LOGGER = logging.getLogger(__name__)


class AreaResolverCapability(Capability):
    """
    Resolve area/floor names from user input.
    
    Combines:
    - Fast path: Exact match, global keywords
    - Fuzzy match: Canonicalized name comparison, HA aliases
    - LLM fallback: For complex mappings
    """

    name = "area_resolver"
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
""",
        "schema": {
            "type": "object",
            "properties": {
                "match": {"type": ["string", "null"]},
            },
            "required": ["match"],
        },
    }

    # --- Direct Registry Lookup Methods ---
    
    def find_area(self, area_name: Optional[str]):
        """Find area by name with fuzzy matching.
        
        Args:
            area_name: User-provided area name
            
        Returns:
            AreaEntry or None. Returns None for global keywords like 'Haus'.
        """
        if not area_name:
            return None
        
        # Check for global keywords - return None to trigger all-domain lookup
        area_lower = area_name.lower()
        if any(gk in area_lower for gk in GLOBAL_KEYWORDS):
            _LOGGER.debug("[AreaResolver] Global scope detected: '%s' → all entities", area_name)
            return None
            
        area_reg = ar.async_get(self.hass)
        needle = canonicalize(area_name)
        areas = area_reg.async_list_areas()
        
        # First pass: exact name match
        for a in areas:
            canon_name = canonicalize(a.name or "")
            if canon_name == needle:
                return a
        
        # Second pass: check HA aliases (e.g., "S-Zimmer" alias for "Esszimmer")
        for a in areas:
            aliases = getattr(a, "aliases", None) or set()
            for alias in aliases:
                if canonicalize(alias) == needle:
                    _LOGGER.debug("[AreaResolver] Area alias match: '%s' → '%s'", area_name, a.name)
                    return a
        
        # Third pass: partial match (name contains needle or vice versa)
        for a in areas:
            canon_name = canonicalize(a.name or "")
            if needle in canon_name or canon_name in needle:
                _LOGGER.debug("[AreaResolver] Area partial match: '%s' → '%s'", area_name, a.name)
                return a
        
        return None

    def find_floor(self, floor_name: str):
        """Find floor by name with alias resolution and fuzzy matching.
        
        Args:
            floor_name: User-provided floor name
            
        Returns:
            FloorEntry or None
        """
        if not floor_name:
            return None
        
        floor_reg = fr.async_get(self.hass)
        floors = list(floor_reg.async_list_floors())
        needle = canonicalize(floor_name)
        
        # Expand needle to include common German floor aliases
        search_terms = {needle}
        if needle in FLOOR_ALIASES_DE:
            search_terms.update(FLOOR_ALIASES_DE[needle])
        
        # First pass: exact name match
        for floor in floors:
            floor_canon = canonicalize(floor.name)
            if floor_canon in search_terms:
                _LOGGER.debug("[AreaResolver] Floor match: '%s' → '%s'", floor_name, floor.name)
                return floor
        
        # Second pass: check HA registered floor aliases
        for floor in floors:
            aliases = getattr(floor, "aliases", None) or set()
            for alias in aliases:
                if canonicalize(alias) in search_terms or needle == canonicalize(alias):
                    _LOGGER.debug("[AreaResolver] Floor HA alias match: '%s' → '%s'", floor_name, floor.name)
                    return floor
        
        # Third pass: partial match (name contains search term or vice versa)
        for floor in floors:
            floor_canon = canonicalize(floor.name)
            for term in search_terms:
                if term in floor_canon or floor_canon in term:
                    _LOGGER.debug("[AreaResolver] Floor partial match: '%s' → '%s'", floor_name, floor.name)
                    return floor
        
        _LOGGER.debug("[AreaResolver] No floor found for '%s'", floor_name)
        return None

    # --- LLM-based Resolution (for complex cases) ---
    
    async def run(
        self, 
        user_input, 
        search_text: str = None,
        area_name: str = None,  # Convenience alias for search_text
        mode: str = "area",  # "area" or "floor"
        **_: Any
    ) -> Dict[str, Any]:
        """Resolve area/floor name with LLM fallback.
        
        Args:
            user_input: ConversationInput
            search_text: Text to search for
            area_name: Alias for search_text
            mode: "area" or "floor"
            
        Returns:
            Dict with "match" key (string or None)
        """
        # Support both search_text and area_name kwargs
        query = search_text or area_name
        if not query and user_input is not None:
            query = getattr(user_input, "text", "") or ""
        text = (query or "").strip()
        
        if not text:
            return {"match": None}

        # Check for global keywords locally (faster than LLM)
        if text.lower() in GLOBAL_KEYWORDS:
            return {"match": "GLOBAL"}

        # Try fast path first
        if mode == "floor":
            floor_obj = self.find_floor(text)
            if floor_obj:
                return {"match": floor_obj.name}
            # Load candidates for LLM
            floor_reg = fr.async_get(self.hass)
            candidates = [f.name for f in floor_reg.async_list_floors() if f.name]
        else:
            area_obj = self.find_area(text)
            if area_obj:
                return {"match": area_obj.name}
            # Load candidates for LLM
            area_reg = ar.async_get(self.hass)
            candidates = [a.name for a in area_reg.async_list_areas() if a.name]

        if not candidates:
            return {"match": None}

        # LLM fallback for complex cases (synonyms, abbreviations)
        payload = {
            "user_query": text,
            "candidates": candidates,
        }

        data = await self._safe_prompt(self.PROMPT, payload)

        if not isinstance(data, dict):
            # LLM failed - return unknown area with candidates for user disambiguation
            _LOGGER.debug("[AreaResolver] LLM failed, returning unknown area: %s", text)
            return {
                "match": None,
                "unknown_area": text,
                "candidates": candidates,
            }

        matched = data.get("match")
        
        if matched == "GLOBAL":
            return {"match": "GLOBAL"}

        if matched and matched in candidates:
            _LOGGER.debug("[AreaResolver] LLM mapped '%s' → '%s' (mode=%s)", text, matched, mode)
            return {"match": matched}

        # LLM returned no match - return unknown area for user disambiguation
        _LOGGER.debug("[AreaResolver] No match for '%s', returning for user disambiguation", text)
        return {
            "match": None,
            "unknown_area": text,
            "candidates": candidates,
        }

    async def learn_area_alias(self, alias: str, area_name: str) -> None:
        """Learn an area alias after user confirms.
        
        Args:
            alias: The unknown text user said (e.g., "Ki-Bad")
            area_name: The actual area name (e.g., "Kinder Badezimmer")
        """
        from .memory import MemoryCapability
        memory = MemoryCapability(self.hass, self._config)
        await memory.learn_area_alias(alias.lower(), area_name)
        _LOGGER.info("[AreaResolver] Learned area alias: '%s' → '%s'", alias, area_name)


