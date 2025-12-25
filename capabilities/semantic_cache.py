"""Semantic Command Cache Capability.

Uses external add-on for cache lookup (vector search + reranking).
Stores new user-learned entries locally, which the add-on watches and reloads.

Config options:
    cache_enabled: Enable semantic cache (default: True)
    reranker_ip: Add-on hostname (default: localhost)
    reranker_port: Add-on port (default: 9876)
    reranker_enabled: Use cache lookup (default: True)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

from .base import Capability
from ..utils.semantic_cache_types import (
    CacheEntry,
    MIN_CACHE_WORDS,
    DEFAULT_MAX_ENTRIES,
)
from ..utils.german_utils import normalize_for_cache

_LOGGER = logging.getLogger(__name__)

IMPLICIT_PATTERNS = [
    r"\bzu\s+dunkel\b",
    r"\bzu\s+hell\b",
    r"\bzu\s+kalt\b",
    r"\bzu\s+warm\b",
    r"\bzu\s+heiß\b",
    r"\bes\s+ist\s+(dunkel|hell)\b",
    r"\b(dunkel|hell)\s+hier\b",
]


class SemanticCacheCapability(Capability):
    """Semantic cache with add-on lookup and local storage for learning."""

    name = "semantic_cache"
    description = "Semantic caching via add-on API with local learning"

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._cache: List[CacheEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._cache_file = os.path.join(
            hass.config.path(".storage"), "multistage_assist_semantic_cache.json"
        )
        self._stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_errors": 0,
        }
        self._loaded = False
        self._embedding_dim: Optional[int] = None

        # Add-on config - handles both lookup and embeddings
        self.addon_ip = config.get("reranker_ip", "localhost")
        self.addon_port = config.get("reranker_port", 9876)
        self.reranker_enabled = config.get("reranker_enabled", True)

        # Cache settings
        self.max_entries = config.get("cache_max_entries", DEFAULT_MAX_ENTRIES)
        self.enabled = config.get("cache_enabled", True)

        _LOGGER.info(
            "[SemanticCache] Configured: enabled=%s, add-on=%s:%s",
            self.enabled,
            self.addon_ip,
            self.addon_port,
        )

    async def async_startup(self):
        """Initialize cache at integration startup."""
        if not self.enabled:
            return
        
        # Load existing user-learned cache (for store() duplicate check)
        await self._load_cache()
        
        # Initialize anchor cache via builder
        from .semantic_cache_builder import SemanticCacheBuilder
        builder = SemanticCacheBuilder(
            self.hass, 
            self.config, 
            self._get_embedding, 
            self._normalize_numeric_value
        )
        
        # Try to load existing anchors
        success, anchors = await builder.load_anchor_cache()
        if success and anchors:
            _LOGGER.info("[SemanticCache] Loaded %d anchors from cache", len(anchors))
        else:
            # Generate anchors in background (non-blocking)
            _LOGGER.info("[SemanticCache] Generating anchors in background...")
            asyncio.create_task(self._generate_anchors_background(builder))
        
        _LOGGER.info("[SemanticCache] Startup complete: %d user entries loaded", len(self._cache))

    async def _generate_anchors_background(self, builder):
        """Generate anchors in background task."""
        try:
            anchors = await builder.generate_anchors()
            if anchors:
                await builder.save_anchor_cache(anchors)
                _LOGGER.info("[SemanticCache] Generated %d anchors", len(anchors))
        except Exception as e:
            _LOGGER.error("[SemanticCache] Anchor generation failed: %s", e)

    def _addon_url(self, endpoint: str) -> str:
        """Get add-on API URL for given endpoint."""
        return f"http://{self.addon_ip}:{self.addon_port}{endpoint}"

    def _should_bypass(self, text: str) -> bool:
        """Check if query should bypass cache lookup.
        
        Only implicit patterns (e.g., "zu dunkel") bypass cache
        since they need LLM interpretation.
        
        Duration patterns (e.g., "für 5 Minuten") are handled via
        cache normalization and extraction in stage1_cache.
        """
        text_lower = text.lower()
        for pattern in IMPLICIT_PATTERNS:
            if re.search(pattern, text_lower):
                _LOGGER.debug("[SemanticCache] Bypass pattern detected: %s", text[:50])
                return True
        return False


    async def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        """Lookup command via add-on API.
        
        Args:
            text: User command text
            
        Returns:
            Dict with {intent, entity_ids, slots, score, ...} or None
        """
        if not self.enabled or not self.reranker_enabled:
            return None

        # Check bypass patterns locally (faster than API call)
        if self._should_bypass(text):
            return None

        self._stats["total_lookups"] += 1

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._addon_url("/lookup"),
                    json={"query": text},
                ) as resp:
                    if resp.status != 200:
                        _LOGGER.warning(
                            "[SemanticCache] Lookup API returned %d", resp.status
                        )
                        self._stats["api_errors"] += 1
                        return None
                    data = await resp.json()
        except asyncio.TimeoutError:
            _LOGGER.warning("[SemanticCache] Lookup API timeout")
            self._stats["api_errors"] += 1
            return None
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Lookup API error: %s", e)
            self._stats["api_errors"] += 1
            return None

        if not data.get("found"):
            self._stats["cache_misses"] += 1
            _LOGGER.debug("[SemanticCache] MISS: %s (score=%.2f)", text[:40], data.get("score", 0))
            return None

        self._stats["cache_hits"] += 1
        
        result = {
            "intent": data["intent"],
            "entity_ids": data.get("entity_ids", []),
            "slots": data.get("slots", {}),
            "score": data.get("score", 0),
            "original_text": data.get("original_text", ""),
            "source": "anchor" if data.get("is_anchor") else "learned",
        }
        
        _LOGGER.info(
            "[SemanticCache] HIT: '%s' → %s (score=%.2f, source=%s)",
            text[:40], result["intent"], result["score"], result["source"]
        )
        
        return result

    async def store(
        self,
        text: str,
        intent: str,
        entity_ids: List[str],
        slots: Dict[str, Any],
        required_disambiguation: bool = False,
        disambiguation_options: Optional[Dict[str, str]] = None,
        verified: bool = True,
        is_disambiguation_response: bool = False,
    ):
        """Cache a successful command resolution.
        
        Stores to disk; add-on watches file and reloads automatically.
        Only call this AFTER verified successful execution.
        """
        if not self.enabled:
            return

        if not verified:
            _LOGGER.debug("[SemanticCache] SKIP: unverified command")
            return

        if is_disambiguation_response:
            _LOGGER.info("[SemanticCache] SKIP disambig response: '%s'", text[:40])
            return

        word_count = len(text.strip().split())
        if word_count < MIN_CACHE_WORDS:
            _LOGGER.info(
                "[SemanticCache] SKIP too short (%d words): '%s'", word_count, text[:40]
            )
            return

        # Skip non-repeatable commands
        if intent in (
            "HassCalendarCreate",
            "HassCreateEvent",
            "HassTimerSet",
            "HassStartTimer",
            "TemporaryControl",
            "DelayedControl",
        ):
            _LOGGER.debug("[SemanticCache] SKIP: non-repeatable intent %s", intent)
            return

        # Normalize text for generalized matching
        text_norm, _ = self._normalize_numeric_value(text)
        if text_norm != text:
            _LOGGER.debug("[SemanticCache] Generalized: '%s' → '%s'", text, text_norm)
            text = text_norm

        await self._load_cache()

        embedding = await self._get_embedding(text)
        if embedding is None:
            return

        # Check for near-duplicate
        if self._embeddings_matrix is not None and len(self._cache) > 0:
            similarities = self._cosine_similarity(embedding, self._embeddings_matrix)
            best_idx = int(np.argmax(similarities))
            if similarities[best_idx] > 0.95:
                _LOGGER.debug(
                    "[SemanticCache] Updating existing entry (%.3f similarity)",
                    similarities[best_idx],
                )
                self._cache[best_idx].hits += 1
                self._cache[best_idx].last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")
                await self._save_cache()
                return

        # Filter out runtime-computed values
        filtered_slots = {
            k: v for k, v in (slots or {}).items()
            if k not in ("brightness", "_prerequisites")
        }

        entry = CacheEntry(
            text=text,
            embedding=embedding.tolist(),
            intent=intent,
            entity_ids=entity_ids,
            slots=filtered_slots,
            required_disambiguation=required_disambiguation,
            disambiguation_options=disambiguation_options,
            hits=1,
            last_hit=time.strftime("%Y-%m-%dT%H:%M:%S"),
            verified=verified,
        )

        self._cache.append(entry)

        if self._embeddings_matrix is None:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, embedding])

        # LRU eviction
        if len(self._cache) > self.max_entries:
            self._cache.sort(key=lambda e: e.last_hit, reverse=True)
            removed = self._cache[self.max_entries:]
            self._cache = self._cache[:self.max_entries]
            self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            _LOGGER.debug("[SemanticCache] Evicted %d old entries", len(removed))

        await self._save_cache()

        _LOGGER.info(
            "[SemanticCache] Stored: '%s' → %s [%s]",
            text[:40],
            intent,
            entity_ids[0] if entity_ids else "?",
        )

    # --- Helper Methods ---

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text via add-on /embed/text API."""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._addon_url("/embed/text"),
                    json={"text": text},
                ) as resp:
                    if resp.status != 200:
                        _LOGGER.warning("[SemanticCache] Embed API returned %d", resp.status)
                        return None
                    data = await resp.json()
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    if self._embedding_dim is None:
                        self._embedding_dim = data.get("dim", len(embedding))
                    return embedding
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Embed error: %s", e)
            return None

    async def _load_cache(self):
        """Load user-learned cache from disk."""
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

            for item in data.get("entries", []):
                try:
                    entry = CacheEntry(**item)
                    self._cache.append(entry)
                except Exception:
                    continue

            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
                self._embedding_dim = self._embeddings_matrix.shape[1]

            self._loaded = True
            _LOGGER.info("[SemanticCache] Loaded %d user entries", len(self._cache))
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Failed to load cache: %s", e)
            self._loaded = True

    async def _save_cache(self):
        """Persist cache to disk. Add-on watches and reloads automatically."""
        # Only save user-learned entries (not generated/anchors)
        user_entries = [e for e in self._cache if not getattr(e, 'generated', False)]

        data = {
            "version": 5,
            "entries": [asdict(e) for e in user_entries],
            "stats": self._stats,
        }

        try:
            def _write():
                with open(self._cache_file, "w") as f:
                    json.dump(data, f, indent=2)

            await self.hass.async_add_executor_job(_write)
        except Exception as e:
            _LOGGER.error("[SemanticCache] Failed to save cache: %s", e)

    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and cached embeddings."""
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-10)
        return np.dot(matrix_norm, query_norm)

    def _normalize_numeric_value(self, text: str) -> tuple:
        """Normalize numeric values for generalized cache matching.
        
        Returns: (normalized_text, extracted_values)
        """
        return normalize_for_cache(text)


    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            "user_entries": len(self._cache),
            "enabled": self.enabled,
            "addon_url": f"{self.reranker_ip}:{self.reranker_port}",
        }
