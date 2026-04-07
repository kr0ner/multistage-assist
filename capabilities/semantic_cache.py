"""Semantic Command Cache Capability.

Uses external add-on for cache lookup (vector search).
Stores new user-learned entries locally, which the add-on watches and reloads.

Config options:
    cache_enabled: Enable semantic cache (default: True)
    cache_addon_ip: Add-on hostname (default: 192.168.178.2)
    cache_addon_port: Add-on port (default: 9876)
    cache_max_entries: Max local entries before LRU eviction
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set

import aiohttp
import numpy as np

from .base import Capability
from ..utils.german_utils import canonicalize, normalize_for_cache
from ..utils.semantic_cache_types import (
    CacheEntry,
    MIN_CACHE_WORDS,
    DEFAULT_MAX_ENTRIES,
)

_LOGGER = logging.getLogger(__name__)

class SemanticCacheCapability(Capability):
    """Semantic cache with add-on lookup and local storage for learning."""

    name = "semantic_cache"
    description = "Optimizes performance by caching and retrieving successful command resolutions. Utilizes a tiered approach: 1. Exact anchor match 2. Local fuzzy vector search (Learned entries) 3. Remote add-on lookup with reranking. Manages persistent storage of user-learned patterns and background anchor generation for system self-scaling."

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._cache: List[CacheEntry] = []
        self._anchor_texts: Set[str] = set() 
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._cache_file = os.path.join(
            hass.config.path(".storage"), "multistage_assist_semantic_cache.json"
        )
        self._stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_errors": 0,
            "anchor_count": 0,
            "anchor_escalations": 0,
            "real_entries": 0,
        }
        self._loaded = False
        self._embedding_dim: Optional[int] = None

        from ..const import CONF_CACHE_ADDON_IP, CONF_CACHE_ADDON_PORT, DEFAULT_CACHE_ADDON_HOST
        
        # Add-on config - handles lookup and embedding
        self.addon_ip = config.get(CONF_CACHE_ADDON_IP) or DEFAULT_CACHE_ADDON_HOST
        self.addon_port = config.get(CONF_CACHE_ADDON_PORT) or 9876
        
        # Mirroring for stats/internal compatibility
        self.embedding_ip = self.addon_ip
        self.embedding_port = self.addon_port
        self.embedding_model = "addon-integrated"
        
        # Local state
        self.enabled = config.get("cache_enabled", True)
        self.max_entries = config.get("cache_max_entries", DEFAULT_MAX_ENTRIES)

        _LOGGER.info(
            "[SemanticCache] Configured: enabled=%s, add-on=%s:%s (Add-on Only Mode)",
            self.enabled,
            self.addon_ip,
            self.addon_port,
        )

    async def async_startup(self):
        """Initialize cache at integration startup."""
        if not self.enabled:
            return
        
        await self._load_cache()
        self._stats["real_entries"] = len(self._cache)
        
        from ..utils.semantic_cache_builder import SemanticCacheBuilder
        builder = SemanticCacheBuilder(
            self.hass, 
            self.config, 
            self._get_embedding, 
            self._normalize_numeric_value,
            batch_embedding_func=self.async_batch_embed
        )
        
        success, anchors = await builder.load_anchor_cache()
        if success:
            self._handle_anchors_loaded(anchors)
        else:
            _LOGGER.info("[SemanticCache] Generating anchors in background...")
            asyncio.create_task(self._generate_anchors_background(builder))
        
        _LOGGER.info("[SemanticCache] Startup complete: %d entries in matrix", len(self._cache))

    def _handle_anchors_loaded(self, anchors: list[CacheEntry]):
        """Merge anchors into local cache for shared fuzzy search."""
        from ..utils.german_utils import normalize_for_cache
        self._stats["anchor_count"] = len(anchors)
        self._anchor_texts = {a.text for a in anchors}
        
        self._cache = [e for e in self._cache if not getattr(e, 'generated', False)]
        self._cache.extend(anchors)
        
        if self._cache:
            valid_embeddings = [np.array(e.embedding, dtype=np.float32) for e in self._cache if e.embedding]
            if valid_embeddings:
                self._embeddings_matrix = np.vstack(valid_embeddings)
                norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
                self._embeddings_matrix = self._embeddings_matrix / (norms + 1e-10)
                self._embedding_dim = self._embeddings_matrix.shape[1]
            
        self._loaded = True
        _LOGGER.info("[SemanticCache] Matrix built: %d entries (%d user, %d anchors)", 
                     len(self._cache), self._stats["real_entries"], len(anchors))

    async def _generate_anchors_background(self, builder):
        """Generate anchors in background task."""
        try:
            anchors = await builder.generate_anchors()
            if anchors:
                await builder.save_anchor_cache(anchors)
                self._handle_anchors_loaded(anchors)
        except Exception as e:
            _LOGGER.error("[SemanticCache] Anchor generation failed: %s", e)

    def _addon_url(self, endpoint: str) -> str:
        """Get add-on API URL."""
        return f"http://{self.addon_ip}:{self.addon_port}{endpoint}"

    def _embedding_url(self, path: str) -> str:
        """Get embedding API URL."""
        return f"http://{self.addon_ip}:{self.addon_port}{path}"

    async def async_batch_embed(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Generate embeddings for a batch of texts using the addon."""
        if not self.enabled:
            return None

        url = self._embedding_url("/embed")
        entries = [{"text": t, "intent": "none", "entity_ids": [], "slots": {}} for t in texts]
        payload = {"entries": entries}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as resp:
                    if resp.status != 200:
                        _LOGGER.warning("[SemanticCache] Batch embed failed: %d", resp.status)
                        return None
                    data = await resp.json()
                    return [np.array(entry["embedding"], dtype=np.float32) for entry in data.get("entries", [])]
        except Exception as e:
            _LOGGER.error("[SemanticCache] Batch embed error: %s", e)
            return None

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for normalized text using the addon's /embed/text endpoint."""
        url = self._embedding_url("/embed/text")
        payload = {"text": text.strip()} # Already lower/normed from normalize_for_cache

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=15) as resp:
                    if resp.status != 200:
                        _LOGGER.warning("[SemanticCache] Add-on embedding failed: %d", resp.status)
                        return None
                    data = await resp.json()
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    if self._embedding_dim is None:
                        self._embedding_dim = len(embedding)
                    return embedding
        except Exception as e:
            _LOGGER.error("[SemanticCache] Add-on embedding error: %s", e)
            return None

    async def lookup(self, query_text: str, return_anchors: bool = False) -> Optional[Dict]:
        """Lookup query in semantic cache."""
        if not self.enabled:
            return None

        self._stats["total_lookups"] += 1
        from ..utils.german_utils import map_area_alias
        
        # 0. Prep Input: Alias mapping (e.g. "Bad" -> "Badezimmer")
        query_text = map_area_alias(query_text)
        
        # 1. Complete Normalization (Umlauts, centroids, noise-stripping)
        # Production and Test Anchor generation now use this SAME function.
        query_norm, extracted = self._normalize_numeric_value(query_text)

        from ..const import CONF_VECTOR_THRESHOLD, EXPERT_DEFAULTS
        HIT_THRESHOLD = self.config.get(CONF_VECTOR_THRESHOLD, EXPERT_DEFAULTS[CONF_VECTOR_THRESHOLD])
        await self._load_cache()

        # 2. Local Exact Anchor Check (Fastest)
        if query_norm in self._anchor_texts:
            if not return_anchors:
                self._stats["anchor_escalations"] += 1
                return None
            for entry in self._cache:
                if entry.text == query_norm:
                    return {
                        "intent": entry.intent,
                        "entity_ids": entry.entity_ids,
                        "slots": self._denormalize_slots(entry.slots, extracted),
                        "score": 1.0, "source": "anchor", "original_text": entry.text
                    }

        # 2. Local Fuzzy Check (Learned entries)
        if self._cache and self._embeddings_matrix is not None:
            query_emb = await self._get_embedding(query_norm)
            if query_emb is not None:
                similarities = self._cosine_similarity(query_emb, self._embeddings_matrix)
                if len(similarities) > 0:
                    top_indices = np.argsort(similarities)[-5:][::-1]
                    for idx in top_indices:
                        score = float(similarities[idx])
                        if score < HIT_THRESHOLD: continue
                        candidate = self._cache[idx]
                        if candidate.text == "test_text": continue
                        if not self._verify_match_safety(query_norm, candidate): continue
                        
                        if getattr(candidate, "generated", False) and not return_anchors:
                            self._stats["anchor_escalations"] += 1
                            return None

                        self._stats["cache_hits"] += 1
                        candidate.hits += 1
                        candidate.last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")
                        self.hass.async_create_task(self._save_cache())
                        
                        return {
                            "intent": candidate.intent,
                            "entity_ids": candidate.entity_ids,
                            "slots": self._denormalize_slots(candidate.slots, extracted),
                            "score": score, "source": "learned", "original_text": candidate.text
                        }

        # 3. Remote Lookup
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(self._addon_url("/lookup"), json={"query": query_norm, "top_k": 5}) as resp:
                    if resp.status != 200:
                        self._stats["api_errors"] += 1
                        return None
                    data = await resp.json()
        except Exception:
            self._stats["api_errors"] += 1
            return None

        if not data.get("found") or data.get("score", 0) < HIT_THRESHOLD:
            self._stats["cache_misses"] += 1
            return None

        matches = data.get("matches", [data])
        above_threshold = [m for m in matches if m.get("score", 0) >= HIT_THRESHOLD]
        if not above_threshold: return None
        
        best = above_threshold[0]
        if best.get("text") == "test_text": return None

        entry_text_canon = canonicalize(best.get("text", ""))
        is_anchor = entry_text_canon in self._anchor_texts or best.get("source") == "anchor" or best.get("generated") is True

        if is_anchor and not return_anchors:
            self._stats["anchor_escalations"] += 1
            return None

        if not self._verify_match_safety(query_norm, best):
            return None

        result = {
            "intent": best.get("intent"),
            "entity_ids": best.get("entity_ids", []),
            "slots": self._denormalize_slots(best.get("slots", {}), extracted),
            "score": best.get("score", 0),
            "original_text": best.get("original_text", ""),
            "source": "learned",
            "ambiguous_matches": above_threshold if len(above_threshold) > 1 else None,
        }
        
        if len(above_threshold) > 1 and result["intent"] == "HassGetState":
            all_ids = []
            for m in above_threshold: all_ids.extend(m.get("entity_ids", []))
            result["entity_ids"] = list(set(all_ids))
            if "ambiguous_matches" in result:
                result.pop("ambiguous_matches")

        self._stats["cache_hits"] += 1
        return result

    async def store(self, text: str, intent: str, entity_ids: List[str], slots: Dict[str, Any], 
                    verified: bool = True, is_disambiguation_response: bool = False,
                    required_disambiguation: bool = False, disambiguation_options: Optional[Dict[str, str]] = None):
        """Cache a successful command resolution."""
        if not self.enabled or not verified or is_disambiguation_response:
            return

        if text in self._anchor_texts and not any(e.text == text for e in self._cache):
             return

        if len(text.strip().split()) < MIN_CACHE_WORDS:
            return

        if intent in ("HassCalendarCreate", "HassTimerSet", "HassStartTimer", "TemporaryControl"):
            return

        text_norm, _ = self._normalize_numeric_value(text)
        if text_norm in self._anchor_texts: return
        text = text_norm

        await self._load_cache()
        embedding = await self._get_embedding(text)
        if embedding is None: return

        if self._embeddings_matrix is not None and len(self._cache) > 0:
            similarities = self._cosine_similarity(embedding, self._embeddings_matrix)
            if np.max(similarities) > 0.95:
                idx = int(np.argmax(similarities))
                self._cache[idx].hits += 1
                self._cache[idx].last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")
                await self._save_cache()
                return

        entry = CacheEntry(
            text=text, embedding=embedding.tolist(), intent=intent, entity_ids=entity_ids,
            slots={k: v for k, v in (slots or {}).items() if k not in ("brightness", "_prerequisites")},
            required_disambiguation=required_disambiguation,
            disambiguation_options=disambiguation_options,
            hits=1, last_hit=time.strftime("%Y-%m-%dT%H:%M:%S"), verified=True
        )

        self._cache.append(entry)
        if self._embeddings_matrix is None:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, embedding])

        if len(self._cache) > self.max_entries:
            self._cache.sort(key=lambda e: e.last_hit, reverse=True)
            self._cache = self._cache[:self.max_entries]
            self._embeddings_matrix = np.array([e.embedding for e in self._cache])

        await self._save_cache()

    async def _load_cache(self):
        """Load user-learned cache from disk."""
        if self._loaded or not os.path.exists(self._cache_file):
            self._loaded = True
            return
        try:
            def _read():
                with open(self._cache_file, "r") as f: return json.load(f)
            data = await self.hass.async_add_executor_job(_read)
            for item in data.get("entries", []):
                try: self._cache.append(CacheEntry(**item))
                except: continue
            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
                self._embedding_dim = self._embeddings_matrix.shape[1]
            self._loaded = True
        except Exception: self._loaded = True

    async def _save_cache(self):
        """Persist cache to disk."""
        user_entries = [e for e in self._cache if not getattr(e, 'generated', False)]
        data = {"version": 5, "entries": [asdict(e) for e in user_entries], "stats": self._stats}
        try:
            def _write():
                with open(self._cache_file, "w") as f: json.dump(data, f, indent=2)
            await self.hass.async_add_executor_job(_write)
        except Exception as e: _LOGGER.error("[SemanticCache] Save failed: %s", e)

    def _cosine_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix is None or len(matrix) == 0: return np.array([], dtype=np.float32)
        q = query.flatten()
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        m_norm = matrix / (m_norms + 1e-10)
        return np.dot(m_norm, q_norm)

    def _normalize_numeric_value(self, text: str) -> tuple:
        from ..utils.german_utils import normalize_for_cache
        return normalize_for_cache(text)

    def _denormalize_slots(self, slots: Dict[str, Any], extracted: List[Any]) -> Dict[str, Any]:
        if not extracted: return slots.copy()
        new_slots, vals = slots.copy(), list(extracted)
        placeholders = {10, 20, 50, "10 Uhr"}
        for k, v in new_slots.items():
            if v in placeholders and vals: new_slots[k] = vals.pop(0)
        return new_slots

    def _verify_match_safety(self, query_norm: str, entry: Any) -> bool:
        """Harden Principle 3 (Intent Separation) via explicit keyword safety check.
        
        This last-mile safety check prevents cross-intent hits where the embedding
        model might be uncertain (e.g., TurnOn vs TurnOff).
        """
        # query_norm is already canonicalized via normalize_for_cache
        query_canon = query_norm
        
        # Handle both CacheEntry objects and dicts from remote API
        intent = entry.intent if hasattr(entry, "intent") else entry.get("intent")
        entry_slots = (entry.slots if hasattr(entry, "slots") else entry.get("slots", {})) or {}

        # 1. Question vs Command separation
        # Questions typically start with these or contain status-oriented verbs
        question_keywords = ["ist ", "sind ", "brennt ", "leuchtet ", "status", "wie ", "wo ", "welche "]
        is_query_question = any(q in query_canon for q in question_keywords) or query_canon.endswith("?")
        
        if is_query_question and intent not in ["HassGetState", "HassTimerStatus"]:
            return False
        if not is_query_question and intent == "HassGetState":
            # Command phrasing hitting a question intent
            return False

        # 2. Opposite Intent Blocking (Principle 5)
        # TurnOn keywords: an, oeffne, hoch, heller, mach
        # TurnOff keywords: aus, schliesse, runter, zu, dunkler, mach
        
        ON_INDICATORS = {" an", "oeffne", " hoch", "heller", "helligkeit erhoehen"}
        OFF_INDICATORS = {" aus", "schliesse", " runter", " zu", "dunkler", "helligkeit verringern"}

        if intent in ("HassTurnOn", "HassLightSet"):
            if any(off in query_canon for off in OFF_INDICATORS):
                _LOGGER.debug("[SemanticCache] Blocked %s: query contains OFF indicator", intent)
                return False
        
        if intent == "HassTurnOff":
            if any(on in query_canon for on in ON_INDICATORS):
                _LOGGER.debug("[SemanticCache] Blocked %s: query contains ON indicator", intent)
                return False

        if intent == "HassOpenCover":
            if any(off in query_canon for off in OFF_INDICATORS):
                _LOGGER.debug("[SemanticCache] Blocked %s: query contains CLOSE indicator", intent)
                return False
        
        if intent == "HassCloseCover":
            if any(on in query_canon for on in ON_INDICATORS):
                _LOGGER.debug("[SemanticCache] Blocked %s: query contains OPEN indicator", intent)
                return False

        # 3. Spatial matching (Principle 2)
        # If cache hit specifies an area, that area name MUST be in the query
        area = entry_slots.get("area")
        if area:
            area_canon = canonicalize(area)
            if area_canon not in query_canon:
                _LOGGER.debug("[SemanticCache] Blocked: Area '%s' missing from query", area)
                return False
                
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return cache performance statistics."""
        return {
            **self._stats, "real_entries": len(self._cache), "enabled": self.enabled,
            "cache_addon_url": f"{self.addon_ip}:{self.addon_port}",
            "embedding_url": f"{self.embedding_ip}:{self.embedding_port}"
        }
