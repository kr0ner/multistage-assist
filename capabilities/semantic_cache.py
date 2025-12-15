"""Semantic Command Cache Capability.

Uses sentence embeddings to find similar previously-executed commands
and bypass LLM resolution for repeated/similar requests.

Model: paraphrase-multilingual-MiniLM-L12-v2 (German-native, ~500MB)
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Capability

_LOGGER = logging.getLogger(__name__)

# Model name
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_MAX_ENTRIES = 200
EMBEDDING_DIM = 384  # MiniLM output dimension


@dataclass
class CacheEntry:
    """A cached command resolution."""
    text: str                          # Original command text
    embedding: List[float]             # 384-dim embedding vector
    intent: str                        # Resolved intent
    entity_ids: List[str]              # Resolved entity IDs
    slots: Dict[str, Any]              # Resolved slots
    required_disambiguation: bool      # True if user had to choose
    disambiguation_options: Optional[Dict[str, str]]  # {entity_id: name} if disambiguation needed
    hits: int                          # Number of times this was reused
    last_hit: str                      # ISO timestamp of last use
    verified: bool                     # True if execution was verified successful


class SemanticCacheCapability(Capability):
    """
    Fast-path resolution using sentence embeddings.
    
    Caches successfully executed commands and finds similar ones
    to bypass LLM calls for repeated requests.
    """
    
    name = "semantic_cache"
    description = "Semantic command caching for fast-path resolution"
    
    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._model = None
        self._model_lock = asyncio.Lock()
        self._cache: List[CacheEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._cache_file = os.path.join(
            hass.config.path(".storage"), 
            "multistage_assist_semantic_cache.json"
        )
        self._stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._loaded = False
        
        # Config
        self.threshold = config.get("cache_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        self.max_entries = config.get("cache_max_entries", DEFAULT_MAX_ENTRIES)
    
    async def _load_model(self):
        """Lazy load the sentence transformer model."""
        async with self._model_lock:
            if self._model is not None:
                return self._model
            
            _LOGGER.info("[SemanticCache] Loading embedding model: %s", MODEL_NAME)
            start = time.time()
            
            def _load():
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(MODEL_NAME)
            
            try:
                self._model = await self.hass.async_add_executor_job(_load)
                _LOGGER.info(
                    "[SemanticCache] Model loaded in %.2fs", 
                    time.time() - start
                )
            except Exception as e:
                _LOGGER.error("[SemanticCache] Failed to load model: %s", e)
                return None
            
            return self._model
    
    async def _load_cache(self):
        """Load cache from disk."""
        if self._loaded:
            return
        
        if not os.path.exists(self._cache_file):
            self._loaded = True
            return
        
        try:
            def _read():
                with open(self._cache_file, "r") as f:
                    return json.load(f)
            
            data = await self.hass.async_add_executor_job(_read)
            
            # Parse entries
            for entry_data in data.get("entries", []):
                self._cache.append(CacheEntry(**entry_data))
            
            self._stats = data.get("stats", self._stats)
            
            # Build embeddings matrix for fast similarity search
            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            
            _LOGGER.info(
                "[SemanticCache] Loaded %d cached commands", len(self._cache)
            )
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Failed to load cache: %s", e)
        
        self._loaded = True
    
    async def _save_cache(self):
        """Persist cache to disk."""
        data = {
            "version": 1,
            "entries": [asdict(e) for e in self._cache],
            "stats": self._stats,
        }
        
        try:
            def _write():
                with open(self._cache_file, "w") as f:
                    json.dump(data, f, indent=2)
            
            await self.hass.async_add_executor_job(_write)
        except Exception as e:
            _LOGGER.error("[SemanticCache] Failed to save cache: %s", e)
    
    async def _embed(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text."""
        model = await self._load_model()
        if model is None:
            return None
        
        def _encode():
            return model.encode(text, convert_to_numpy=True)
        
        return await self.hass.async_add_executor_job(_encode)
    
    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all cached embeddings."""
        # Normalize
        query_norm = query / np.linalg.norm(query)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        # Dot product = cosine similarity for normalized vectors
        return np.dot(matrix_norm, query_norm)
    
    async def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Find cached resolution for similar command.
        
        Returns:
            Dict with {intent, entity_ids, slots, score, required_disambiguation, 
                       disambiguation_options} or None if no match.
        """
        await self._load_cache()
        
        if not self._cache or self._embeddings_matrix is None:
            self._stats["cache_misses"] += 1
            return None
        
        self._stats["total_lookups"] += 1
        
        # Get query embedding
        query_emb = await self._embed(text)
        if query_emb is None:
            self._stats["cache_misses"] += 1
            return None
        
        # Find best match
        similarities = self._cosine_similarity(query_emb, self._embeddings_matrix)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        if best_score < self.threshold:
            _LOGGER.debug(
                "[SemanticCache] MISS: best score %.3f < threshold %.3f",
                best_score, self.threshold
            )
            self._stats["cache_misses"] += 1
            return None
        
        # Cache hit!
        entry = self._cache[best_idx]
        entry.hits += 1
        entry.last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        self._stats["cache_hits"] += 1
        
        _LOGGER.info(
            "[SemanticCache] HIT (%.3f): '%s' -> '%s' [%s]",
            best_score, text[:40], entry.intent, entry.entity_ids[0] if entry.entity_ids else "?"
        )
        
        return {
            "intent": entry.intent,
            "entity_ids": entry.entity_ids,
            "slots": entry.slots,
            "score": best_score,
            "required_disambiguation": entry.required_disambiguation,
            "disambiguation_options": entry.disambiguation_options,
            "original_text": entry.text,
        }
    
    async def store(
        self,
        text: str,
        intent: str,
        entity_ids: List[str],
        slots: Dict[str, Any],
        required_disambiguation: bool = False,
        disambiguation_options: Optional[Dict[str, str]] = None,
        verified: bool = True,
    ):
        """
        Cache a successful command resolution.
        
        Only call this AFTER verified successful execution.
        
        Args:
            text: Original command text
            intent: Resolved intent
            entity_ids: Final entity IDs (after disambiguation if any)
            slots: Resolved slots
            required_disambiguation: True if user had to choose
            disambiguation_options: The options shown to user if disambiguation
            verified: True if execution was verified successful
        """
        if not verified:
            _LOGGER.debug("[SemanticCache] Skipping unverified command")
            return
        
        # Skip calendar commands - they don't repeat
        if intent in ("HassCalendarCreate", "HassCreateEvent"):
            return
        
        await self._load_cache()
        
        # Get embedding
        embedding = await self._embed(text)
        if embedding is None:
            return
        
        # Check for duplicate (update existing if very similar)
        if self._embeddings_matrix is not None and len(self._cache) > 0:
            similarities = self._cosine_similarity(embedding, self._embeddings_matrix)
            best_idx = int(np.argmax(similarities))
            if similarities[best_idx] > 0.98:
                # Update existing entry
                _LOGGER.debug(
                    "[SemanticCache] Updating existing entry (%.3f similarity)",
                    similarities[best_idx]
                )
                self._cache[best_idx].hits += 1
                self._cache[best_idx].last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")
                await self._save_cache()
                return
        
        # Create new entry
        entry = CacheEntry(
            text=text,
            embedding=embedding.tolist(),
            intent=intent,
            entity_ids=entity_ids,
            slots=slots,
            required_disambiguation=required_disambiguation,
            disambiguation_options=disambiguation_options,
            hits=1,
            last_hit=time.strftime("%Y-%m-%dT%H:%M:%S"),
            verified=verified,
        )
        
        self._cache.append(entry)
        
        # Update embeddings matrix
        if self._embeddings_matrix is None:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, embedding])
        
        # LRU eviction if needed
        if len(self._cache) > self.max_entries:
            # Sort by last_hit and remove oldest
            self._cache.sort(key=lambda e: e.last_hit, reverse=True)
            removed = self._cache[self.max_entries:]
            self._cache = self._cache[:self.max_entries]
            self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            _LOGGER.debug("[SemanticCache] Evicted %d old entries", len(removed))
        
        await self._save_cache()
        
        _LOGGER.info(
            "[SemanticCache] Stored: '%s' -> %s [%s]",
            text[:40], intent, entity_ids[0] if entity_ids else "?"
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        await self._load_cache()
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": (
                self._stats["cache_hits"] / self._stats["total_lookups"] * 100
                if self._stats["total_lookups"] > 0 else 0
            ),
        }
    
    async def clear(self):
        """Clear all cached entries."""
        self._cache = []
        self._embeddings_matrix = None
        self._stats = {"total_lookups": 0, "cache_hits": 0, "cache_misses": 0}
        await self._save_cache()
        _LOGGER.info("[SemanticCache] Cache cleared")
