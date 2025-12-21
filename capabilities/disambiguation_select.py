"""Disambiguation selection capability with fast paths.

Handles user's follow-up answer to select entity(s) from disambiguation options.
Uses Python fast paths for ordinals, keywords, and fuzzy matching before LLM fallback.
"""

import logging
import re
from typing import Any, Dict, List

from .base import Capability
from ..constants.messages_de import ORDINAL_MAP, ALL_KEYWORDS, NONE_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class DisambiguationSelectCapability(Capability):
    """
    Map the user's follow-up answer to one or more entity_ids from the given candidates.
    
    Uses fast paths before LLM:
    1. Ordinals: "erste", "1.", "nummer 1"
    2. Keywords: "alle", "keine", "beide"
    3. Fuzzy matching on friendly names
    4. LLM fallback for complex cases
    """

    name = "disambiguation_select"
    description = "Select entity_ids from candidates based on user_input."

    PROMPT = {
        "system": """
You select which candidates the user meant. Do not write explanations.

## Input
- user_input: German response (e.g., "Spiegellicht", "erste", "zweite", "alle", "keine").
- input_entities: ordered list of { "entity_id": string, "name": string, "ordinal": integer }.

## Rules
1. Ordinals: The field "ordinal" gives the numeric order for each candidate.
   - "erste" → ordinal = 1
   - "zweite" → ordinal = 2
   - "letzte" → highest ordinal in input_entities
2. Friendly name fuzzy matching:
   - Normalize lowercased names (ignore accents, trim spaces/punctuation)
   - If user_input contains a target that is common in the list, prefer a direct match
3. "alle" → return all entity_ids.
4. "beide" → return all entity_ids if length is two.
5. "keine", "nichts", "nein" → return an empty array.
6. If uncertain or ambiguous → return an empty array.
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, candidates: List[Dict[str, Any]], **_: Any) -> List[str]:
        """Select entities from candidates based on user input.
        
        Args:
            user_input: ConversationInput with user's selection text
            candidates: List of {"entity_id": str, "name": str, "ordinal": int}
        
        Returns:
            List of selected entity_ids
        """
        text = user_input.text.strip().lower()
        
        if not text or not candidates:
            return []

        entity_ids = [c["entity_id"] for c in candidates]
        
        # Fast path 1: Check for "none" keywords
        if self._is_none_selection(text):
            _LOGGER.debug("[DisambiguationSelect] Fast path: 'keine' → []")
            return []
        
        # Fast path 2: Check for "all" keywords  
        if self._is_all_selection(text, len(candidates)):
            _LOGGER.debug("[DisambiguationSelect] Fast path: 'alle/beide' → all %d", len(candidates))
            return entity_ids
        
        # Fast path 3: Ordinal detection
        ordinal = self._detect_ordinal(text)
        if ordinal is not None:
            if ordinal == -1:  # "letzte"
                ordinal = len(candidates)
            
            if 1 <= ordinal <= len(candidates):
                selected = entity_ids[ordinal - 1]
                _LOGGER.debug("[DisambiguationSelect] Fast path: ordinal %d → %s", ordinal, selected)
                return [selected]
        
        # Fast path 4: Fuzzy name match
        fuzzy_match = self._fuzzy_match_name(text, candidates)
        if fuzzy_match:
            _LOGGER.debug("[DisambiguationSelect] Fast path: fuzzy '%s' → %s", text, fuzzy_match)
            return [fuzzy_match]
        
        # LLM fallback for complex cases
        _LOGGER.debug("[DisambiguationSelect] No fast path match, calling LLM")
        raw = await self._safe_prompt(
            self.PROMPT,
            {"user_input": user_input.text, "input_entities": candidates},
            temperature=0.0,
        )
        
        # Result is always a list (schema enforces this)
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, str)]
        return []

    def _is_none_selection(self, text: str) -> bool:
        """Check if user selected 'none'."""
        words = set(text.split())
        return bool(words & NONE_KEYWORDS)
    
    def _is_all_selection(self, text: str, count: int) -> bool:
        """Check if user selected 'all' or 'both'."""
        words = set(text.split())
        if words & {"beide", "beiden", "beides"}:
            return count == 2
        return bool(words & ALL_KEYWORDS)
    
    def _detect_ordinal(self, text: str) -> int | None:
        """Detect ordinal from text. Returns 1-based index or -1 for 'last'."""
        # Check word-based ordinals
        for word in text.split():
            clean = word.rstrip(".,!?")
            if clean in ORDINAL_MAP:
                return ORDINAL_MAP[clean]
        
        # Check numeric patterns: "1", "1.", "nr 1", "nummer 1", "nr. 1"
        patterns = [
            r"^(\d+)\.?$",           # "1" or "1."
            r"^nr\.?\s*(\d+)$",      # "nr 1", "nr. 1"
            r"^nummer\s*(\d+)$",     # "nummer 1"
            r"^die\s+(\d+)\.$",      # "die 1."
        ]
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                return int(match.group(1))
        
        return None
    
    def _fuzzy_match_name(self, text: str, candidates: List[Dict[str, Any]]) -> str | None:
        """Try to fuzzy match user text to a candidate name.
        
        Returns entity_id of best match, or None if no good match.
        """
        # Normalize text for comparison
        text_norm = self._normalize(text)
        
        best_match = None
        best_score = 0
        
        for c in candidates:
            name = c.get("name", "")
            name_norm = self._normalize(name)
            
            # Exact match (normalized)
            if text_norm == name_norm:
                return c["entity_id"]
            
            # User text is contained in name
            if text_norm in name_norm:
                score = len(text_norm) / len(name_norm)
                if score > best_score:
                    best_score = score
                    best_match = c["entity_id"]
            
            # Name is contained in user text
            if name_norm in text_norm:
                score = len(name_norm) / len(text_norm)
                if score > best_score:
                    best_score = score
                    best_match = c["entity_id"]
        
        # Only return if score is high enough to be confident
        if best_score >= 0.5:
            return best_match
        
        return None
    
    def _normalize(self, text: str) -> str:
        """Normalize text for fuzzy matching."""
        text = text.lower().strip()
        # German umlaut normalization
        text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        text = text.replace("ß", "ss")
        # Remove common articles and punctuation
        text = re.sub(r"^(der|die|das|den|dem)\s+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text
