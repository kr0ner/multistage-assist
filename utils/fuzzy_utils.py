"""Fuzzy matching utilities for entity and text matching.

Provides centralized fuzzy matching functionality using rapidfuzz library.
"""

import asyncio
import importlib
import logging
from typing import List, Tuple, Optional

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
