"""Shared types and constants for semantic cache.

This module contains shared data structures and configuration used by both
the cache builder and cache retrieval capabilities.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# Default embedding model (local, 384-dim)
DEFAULT_EMBEDDING_MODEL = "multilingual-minilm"
# Semantic Cache configuration

# Configuration defaults
DEFAULT_VECTOR_THRESHOLD = 0.75  # Must match const.py EXPERT_DEFAULTS[CONF_VECTOR_THRESHOLD]
DEFAULT_VECTOR_TOP_K = 10  # Number of candidates for lookup
DEFAULT_MAX_ENTRIES = 2000  # Must be large enough for all generated anchors + user entries
MIN_CACHE_WORDS = 3


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
