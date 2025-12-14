"""Fuzzy matching utilities for entity and text matching.

Provides centralized fuzzy matching functionality using rapidfuzz library.
"""

import asyncio
import importlib
import logging
from typing import Dict, List, Tuple, Optional

_LOGGER = logging.getLogger(__name__)

# Global cache for rapidfuzz.fuzz module
_fuzz = None


async def get_fuzz():
    """Lazy-load rapidfuzz.fuzz module in executor to avoid blocking.

    Returns:
        rapidfuzz.fuzz module
    """
    global _fuzz
    if _fuzz is not None:
        return _fuzz

    loop = asyncio.get_event_loop()
    _fuzz = await loop.run_in_executor(
        None, lambda: importlib.import_module("rapidfuzz.fuzz")
    )
    _LOGGER.debug("[FuzzyUtils] rapidfuzz.fuzz loaded")
    return _fuzz


async def fuzzy_match_best(
    query: str, candidates: List[str], threshold: int = 70, score_cutoff: int = 0
) -> Optional[Tuple[str, int]]:
    """Find the best fuzzy match from candidates.

    Args:
        query: Search query string
        candidates: List of candidate strings to match against
        threshold: Minimum score to consider a match (0-100)
        score_cutoff: Minimum score for rapidfuzz (default 0)

    Returns:
        Tuple of (best_match, score) if score >= threshold, else None

    Example:
        match, score = await fuzzy_match_best("kitchen", ["küche", "garage"], threshold=70)
        # Returns ("küche", 85) if score >= 70
    """
    if not query or not candidates:
        return None

    fuzz = await get_fuzz()

    best_match = None
    best_score = 0

    for candidate in candidates:
        score = fuzz.ratio(query.lower(), candidate.lower(), score_cutoff=score_cutoff)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        _LOGGER.debug(
            "[FuzzyUtils] Best match for '%s': '%s' (score: %d)",
            query,
            best_match,
            best_score,
        )
        return (best_match, best_score)

    _LOGGER.debug(
        "[FuzzyUtils] No match for '%s' above threshold %d (best: %d)",
        query,
        threshold,
        best_score,
    )
    return None


async def fuzzy_match_all(
    query: str, candidates: List[str], threshold: int = 70
) -> List[Tuple[str, int]]:
    """Find all fuzzy matches above threshold, sorted by score.

    Args:
        query: Search query string
        candidates: List of candidate strings to match against
        threshold: Minimum score to consider a match (0-100)

    Returns:
        List of (match, score) tuples sorted by score descending

    Example:
        matches = await fuzzy_match_all("buro", ["büro", "bureau", "garage"])
        # Returns [("büro", 90), ("bureau", 85)]
    """
    if not query or not candidates:
        return []

    fuzz = await get_fuzz()

    matches = []
    for candidate in candidates:
        score = fuzz.ratio(query.lower(), candidate.lower())
        if score >= threshold:
            matches.append((candidate, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)

    _LOGGER.debug(
        "[FuzzyUtils] Found %d matches for '%s' above threshold %d",
        len(matches),
        query,
        threshold,
    )
    return matches


def fuzzy_match(query: str, candidate: str) -> int:
    """Simple synchronous fuzzy match returning score 0-100.
    
    Uses difflib as fallback if rapidfuzz is not loaded yet.
    
    Args:
        query: First string
        candidate: Second string
        
    Returns:
        Match score 0-100
    """
    if not query or not candidate:
        return 0
    
    # Try rapidfuzz if already loaded
    global _fuzz
    if _fuzz is not None:
        return int(_fuzz.ratio(query.lower(), candidate.lower()))
    
    # Fallback to difflib
    from difflib import SequenceMatcher
    return int(SequenceMatcher(None, query.lower(), candidate.lower()).ratio() * 100)


# Re-export from german_utils for backward compatibility
from .german_utils import GERMAN_ARTICLES, GERMAN_PREPOSITIONS, remove_articles_and_prepositions


def normalize_for_fuzzy(text: str) -> str:
    """Normalize text for fuzzy matching.
    
    Removes German articles and prepositions, lowercases, and strips whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
        
    Example:
        normalize_for_fuzzy("den Keller") -> "keller"
        normalize_for_fuzzy("im Wohnzimmer") -> "wohnzimmer"
    """
    if not text:
        return ""
    
    return remove_articles_and_prepositions(text).lower()


async def fuzzy_match_candidates(
    query: str,
    candidates: List[Dict[str, str]],
    name_key: str = "name",
    id_key: str = "entity_id",
    threshold: int = 70,
    normalize: bool = True,
) -> Optional[str]:
    """Match query against a list of candidate dicts by name then by ID.
    
    This is a common pattern used by timer (devices), calendar, and other
    capabilities that need to match user input to a list of options.
    
    Args:
        query: User's search query
        candidates: List of dicts, each with at least name_key and id_key
        name_key: Key in dict for display name (default: "name")
        id_key: Key in dict for ID/entity_id (default: "entity_id")
        threshold: Minimum match score (0-100)
        normalize: Whether to normalize the query (remove articles)
        
    Returns:
        The id_key value of the best match, or None if no match
        
    Example:
        candidates = [
            {"name": "Family Calendar", "entity_id": "calendar.family"},
            {"name": "Work Calendar", "entity_id": "calendar.work"},
        ]
        result = await fuzzy_match_candidates("family", candidates)
        # Returns "calendar.family"
    """
    if not query or not candidates:
        return None
    
    # Normalize query if requested
    search_query = normalize_for_fuzzy(query) if normalize else query.lower()
    
    # Build lookup dicts
    name_to_id = {c[name_key]: c[id_key] for c in candidates if name_key in c and id_key in c}
    
    # Also try matching by the last part of the ID (after the dot)
    id_part_to_id = {}
    for c in candidates:
        if id_key in c:
            full_id = c[id_key]
            if "." in full_id:
                id_part = full_id.split(".")[-1]
                id_part_to_id[id_part] = full_id
            else:
                id_part_to_id[full_id] = full_id
    
    # Try matching by name first
    match_result = await fuzzy_match_best(
        search_query, list(name_to_id.keys()), threshold=threshold
    )
    if match_result:
        best_match_name, score = match_result
        _LOGGER.debug(
            "[FuzzyUtils] Matched name '%s' to '%s' (score: %d)",
            query, best_match_name, score,
        )
        return name_to_id[best_match_name]
    
    # Try matching by ID part
    match_result = await fuzzy_match_best(
        search_query, list(id_part_to_id.keys()), threshold=threshold
    )
    if match_result:
        best_match_id, score = match_result
        _LOGGER.debug(
            "[FuzzyUtils] Matched ID '%s' to '%s' (score: %d)",
            query, best_match_id, score,
        )
        return id_part_to_id[best_match_id]
    
    _LOGGER.debug("[FuzzyUtils] No match for '%s' in candidates", query)
    return None

