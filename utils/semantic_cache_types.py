"""Shared types and constants for semantic cache.

This module contains shared data structures and configuration used by both
the cache builder and cache retrieval capabilities.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# Default models (4-bit quantized for low RAM)
DEFAULT_EMBEDDING_MODEL = "bge-m3"
# bge-reranker-v2-m3: Better discrimination, ~2.3GB, can run on CPU if GPU OOM
# bge-reranker-base: Smaller (~500MB) but less precise discrimination
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Configuration defaults
# base model score ranges: synonyms ~0.65-0.80, opposites ~0.40, different rooms ~0.35
DEFAULT_RERANKER_THRESHOLD = 0.70  # Fallback for unknown domains
DEFAULT_VECTOR_THRESHOLD = 0.4  # Loose filter for candidate selection
DEFAULT_VECTOR_TOP_K = 10  # Number of candidates to rerank
DEFAULT_MAX_ENTRIES = 200
MIN_CACHE_WORDS = 3

# Per-domain thresholds - optimized through systematic testing
# Testing revealed hit scores cluster around 0.73 for most domains
DOMAIN_THRESHOLDS = {
    "light": 0.73,   # Tested: 9/10 pass at 0.73
    "switch": 0.73,  # Similar to light
    "fan": 0.73,     # Similar to light
    "cover": 0.73,   # Tested: 10/10 pass at 0.73
    "climate": 0.69, # Tested: 7/10 pass at 0.69 (overlapping score ranges)
}


@dataclass
class CacheEntry:
    """A cached command resolution."""

    text: str  # Original command text
    embedding: List[float]  # Embedding vector
    intent: str  # Resolved intent
    entity_ids: List[str]  # Resolved entity IDs
    slots: Dict[str, Any]  # Resolved slots
    required_disambiguation: bool  # True if user had to choose
    disambiguation_options: Optional[
        Dict[str, str]
    ]  # {entity_id: name} if disambiguation
    hits: int  # Number of times reused
    last_hit: str  # ISO timestamp of last use
    verified: bool  # True if execution verified successful
    generated: bool = False  # True = pre-generated entry (from anchors.json)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(**data)
