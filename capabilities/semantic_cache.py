"""Semantic Command Cache Capability.

Uses a two-stage retrieval pipeline:
1. Fast vector search via Ollama embeddings (bge-m3)
2. Precise reranking via CrossEncoder (bge-reranker-base)

This approach correctly distinguishes between:
- Similar commands with different actions ("Licht an" vs "Licht aus")
- Similar commands in different rooms ("Küche Licht" vs "Büro Licht")
- Semantically equivalent variations ("Mach Licht an" vs "Schalte Lampe ein")

Config options:
    cache_enabled: Enable semantic cache (default: True, ~1GB RAM)
    embedding_model: Ollama embedding model (default: bge-m3)
    reranker_model: CrossEncoder model (default: BAAI/bge-reranker-base)
    reranker_enabled: Use reranker for validation (default: True)
    reranker_threshold: Min reranker score for cache hit (default: 0.5)
    vector_search_threshold: Loose filter for candidates (default: 0.4)
    vector_search_top_k: Candidates to rerank (default: 5)
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


# Anchor phrase patterns - 3-tier structure grouped BY DOMAIN
# ⚠️ RULE: 1 ENTRY per domain + intent for each tier
# Each tier uses domain-specific phrasing (e.g., "Öffne" for covers vs "Schalte an" for lights)
#
#   1. AREA scope: "{device} in {area}" → all entities in area
#   2. ENTITY scope: "{device} {entity_name} in {area}" → single entity
#   3. GLOBAL scope: "alle {devices}" → all entities in domain
#
# Format: (pattern, intent, extra_slots)

# AREA-SCOPE patterns: {device} + {area} → resolves to all entities in area
AREA_PHRASE_PATTERNS = {
    "light": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Erhöhe die Helligkeit von {device} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} in {area}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Ist {device} in {area} an", "HassGetState", {}),
    ],
    "cover": [
        ("Öffne {device} in {area}", "HassTurnOn", {}),
        ("Schließe {device} in {area}", "HassTurnOff", {}),
        ("Fahre {device} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Ist {device} in {area} offen", "HassGetState", {}),
    ],
    "climate": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Stelle {device} in {area} auf 21 Grad", "HassClimateSetTemperature", {}),
        ("Wie warm ist es in {area}", "HassGetState", {}),
    ],
    "switch": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Ist {device} in {area} an", "HassGetState", {}),
    ],
    "fan": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Ist {device} in {area} an", "HassGetState", {}),
    ],
    "media_player": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Ist {device} in {area} an", "HassGetState", {}),
    ],
    "automation": [
        ("Aktiviere {device} in {area}", "HassTurnOn", {}),
        ("Deaktiviere {device} in {area}", "HassTurnOff", {}),
        ("Ist {device} in {area} aktiv", "HassGetState", {}),
    ],
}

# ENTITY-SCOPE patterns: {device} + {entity_name} + {area} → single entity
ENTITY_PHRASE_PATTERNS = {
    "light": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Erhöhe die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} {entity_name} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Ist {device} {entity_name} in {area} an", "HassGetState", {}),
    ],
    "cover": [
        ("Öffne {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Schließe {device} {entity_name} in {area}", "HassTurnOff", {}),
        ("Fahre {device} {entity_name} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} {entity_name} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} {entity_name} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Ist {device} {entity_name} in {area} offen", "HassGetState", {}),
    ],
    "climate": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Stelle {device} {entity_name} in {area} auf 21 Grad", "HassClimateSetTemperature", {}),
    ],
    "switch": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Ist {device} {entity_name} in {area} an", "HassGetState", {}),
    ],
    "fan": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
    ],
    "media_player": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
    ],
    "automation": [
        ("Aktiviere {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Deaktiviere {device} {entity_name} in {area}", "HassTurnOff", {}),
    ],
}

# Generic device words by domain
DOMAIN_DEVICE_WORDS = {
    "light": "das Licht",
    "cover": "die Rollläden",
    "climate": "die Heizung",
    "switch": "den Schalter",
    "fan": "den Ventilator",
    "media_player": "den Fernseher",
    "sensor": "den Sensor",
    "automation": "die Automatisierung",
}

# GLOBAL-SCOPE patterns: Domain-wide commands without area restriction
# ⚠️ RULE: 1 ENTRY per domain + intent (same as AREA and ENTITY patterns)
# Format: (text, intent, extra_slots)
GLOBAL_PHRASE_PATTERNS = {
    "light": [
        ("Schalte alle Lichter aus", "HassTurnOff", {}),
        ("Schalte alle Lichter an", "HassTurnOn", {}),
        ("Mach alle Lichter heller", "HassLightSet", {"command": "step_up"}),
        ("Mach alle Lichter dunkler", "HassLightSet", {"command": "step_down"}),
        ("Dimme alle Lichter auf 50 Prozent", "HassLightSet", {"brightness": 50}),
    ],
    "cover": [
        ("Schließe alle Rollläden", "HassTurnOff", {}),  # Close = TurnOff
        ("Öffne alle Rollläden", "HassTurnOn", {}),  # Open = TurnOn
        ("Fahre alle Rollläden weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre alle Rollläden weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle alle Rollläden auf 50 Prozent", "HassSetPosition", {"position": 50}),
    ],
    "switch": [
        ("Schalte alle Schalter aus", "HassTurnOff", {}),
        ("Schalte alle Schalter an", "HassTurnOn", {}),
    ],
    "fan": [
        ("Schalte alle Ventilatoren aus", "HassTurnOff", {}),
        ("Schalte alle Ventilatoren an", "HassTurnOn", {}),
    ],
    "media_player": [
        ("Schalte alle Fernseher aus", "HassTurnOff", {}),
        ("Schalte alle Fernseher an", "HassTurnOn", {}),
    ],
    "automation": [
        ("Deaktiviere alle Automatisierungen", "HassTurnOff", {}),
        ("Aktiviere alle Automatisierungen", "HassTurnOn", {}),
    ],
}


class SemanticCacheCapability(Capability):
    """
    Two-stage semantic cache for fast command resolution.

    Stage 1: Vector search via Ollama embeddings (fast, broad matching)
    Stage 2: CrossEncoder reranking (precise semantic validation)

    RAM Usage: ~1GB total (~700MB embedding via Ollama + ~300MB reranker)
    """

    name = "semantic_cache"
    description = "Two-stage semantic caching with reranker validation"

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
            "reranker_blocks": 0,
            "anchor_escalations": 0,  # Queries escalated due to anchor match
        }
        self._loaded = False
        self._anchors_initialized = False  # True after anchors generated
        self._embedding_dim: Optional[int] = None
        self._reranker = None  # Lazy loaded

        # Embedding config (via Ollama)
        self.embedding_ip = config.get(
            "embedding_ip", config.get("stage1_ip", "localhost")
        )
        self.embedding_port = config.get(
            "embedding_port", config.get("stage1_port", 11434)
        )
        self.embedding_model = config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

        # Reranker config
        # Mode: "local" = use sentence-transformers, "api" = use HTTP API, "auto" = try local first
        self.reranker_mode = config.get("reranker_mode", "auto")
        self.reranker_model = config.get("reranker_model", DEFAULT_RERANKER_MODEL)
        self.reranker_ip = config.get("reranker_ip", "localhost")
        self.reranker_port = config.get("reranker_port", 9876)
        self.reranker_enabled = config.get("reranker_enabled", True)
        self.reranker_threshold = config.get(
            "reranker_threshold", DEFAULT_RERANKER_THRESHOLD
        )
        self._reranker = None  # Lazy loaded for local mode
        self._reranker_mode_resolved = None  # Actual mode after detection

        # Vector search config
        self.vector_threshold = config.get(
            "vector_search_threshold", DEFAULT_VECTOR_THRESHOLD
        )
        self.vector_top_k = config.get("vector_search_top_k", DEFAULT_VECTOR_TOP_K)

        # Cache settings
        self.max_entries = config.get("cache_max_entries", DEFAULT_MAX_ENTRIES)
        # Enabled by default - note: requires ~1GB RAM for embedding + reranker
        self.enabled = config.get("cache_enabled", True)

        _LOGGER.info(
            "[SemanticCache] Configured: enabled=%s, embedding=%s, reranker=%s (mode=%s)",
            self.enabled,
            self.embedding_model,
            (
                self.reranker_model
                if self.reranker_mode != "api"
                else f"{self.reranker_ip}:{self.reranker_port}"
            ),
            self.reranker_mode,
        )

    async def async_startup(self):
        """Initialize cache at integration startup (non-blocking)."""
        if not self.enabled:
            return
        
        # Load existing cache (fast - just read from disk)
        await self._load_cache()
        
        # Try to load pre-verified entries from cache file
        if await self._load_anchor_cache():
            self._anchors_initialized = True
            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            _LOGGER.info("[SemanticCache] Startup complete: %d entries ready", len(self._cache))
        else:
            # No cache - schedule background generation
            _LOGGER.info("[SemanticCache] No cache found, generating in background...")
            self.hass.async_create_task(self._generate_entries_background())

    @property
    def _ollama_url(self) -> str:
        """Get Ollama embeddings API URL."""
        return f"http://{self.embedding_ip}:{self.embedding_port}/api/embeddings"

    @property
    def _reranker_url(self) -> str:
        """Get reranker API URL."""
        return f"http://{self.reranker_ip}:{self.reranker_port}/rerank"

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text via Ollama API."""
        import aiohttp

        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._ollama_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        _LOGGER.error(
                            "[SemanticCache] Ollama API error %d: %s",
                            response.status,
                            error_text[:200],
                        )
                        return None

                    data = await response.json()
                    embedding = data.get("embedding")

                    if embedding is None:
                        _LOGGER.error("[SemanticCache] No embedding in response")
                        return None

                    if self._embedding_dim is None:
                        self._embedding_dim = len(embedding)
                        _LOGGER.debug(
                            "[SemanticCache] Embedding dim: %d", self._embedding_dim
                        )

                    return np.array(embedding, dtype=np.float32)

        except asyncio.TimeoutError:
            _LOGGER.warning("[SemanticCache] Ollama API timeout")
            return None
        except Exception as e:
            _LOGGER.error("[SemanticCache] Failed to get embedding: %s", e)
            return None

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

            for entry_data in data.get("entries", []):
                # Sanitize removed fields to avoid TypeErrors
                entry_data.pop("is_anchor", None)
                self._cache.append(CacheEntry(**entry_data))

            # Merge loaded stats with defaults to ensure all keys exist
            loaded_stats = data.get("stats", {})
            self._stats.update(loaded_stats)

            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
                self._embedding_dim = len(self._cache[0].embedding)

            _LOGGER.info("[SemanticCache] Loaded %d cached commands", len(self._cache))
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Failed to load cache: %s", e)

        self._loaded = True

    async def _load_anchor_cache(self) -> bool:
        """Load anchor cache from disk. Returns True if loaded."""
        anchor_file = os.path.join(
            self.hass.config.path(".storage"), "multistage_assist_anchors.json"
        )
        if not os.path.exists(anchor_file):
            return False

        try:
            def _read():
                with open(anchor_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read)
            
            # Check if anchor cache is compatible
            if data.get("embedding_model") != self.embedding_model:
                _LOGGER.info("[SemanticCache] Anchor model mismatch, regenerating")
                return False
            
            # Load anchors
            for entry_data in data.get("anchors", []):
                # Sanitize removed fields
                entry_data.pop("is_anchor", None)
                self._cache.append(CacheEntry(**entry_data))
            
            _LOGGER.info("[SemanticCache] Loaded %d anchors from cache", 
                        len(data.get("anchors", [])))
            return True
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Failed to load anchor cache: %s", e)
            return False

    async def _save_anchor_cache(self, anchors: list):
        """Save anchor entries to separate cache file."""
        anchor_file = os.path.join(
            self.hass.config.path(".storage"), "multistage_assist_anchors.json"
        )
        
        data = {
            "version": 1,
            "embedding_model": self.embedding_model,
            "anchors": [asdict(e) for e in anchors],
        }

        try:
            def _write():
                with open(anchor_file, "w") as f:
                    json.dump(data, f)

            await self.hass.async_add_executor_job(_write)
            _LOGGER.info("[SemanticCache] Saved %d anchors to cache", len(anchors))
        except Exception as e:
            _LOGGER.error("[SemanticCache] Failed to save anchor cache: %s", e)

    async def _generate_entries_background(self):
        """Generate pre-verified entries in background (non-blocking startup)."""
        try:
            await self._initialize_anchors()
            _LOGGER.info("[SemanticCache] Background generation complete: %d entries", len(self._cache))
        except Exception as e:
            _LOGGER.error("[SemanticCache] Background generation failed: %s", e)

    async def _initialize_anchors(self):
        """
        Generate semantic anchor entries for each domain × intent × area × entity.

        Anchors are cached to disk and only regenerated when embedding model changes.
        First startup takes several minutes, subsequent startups load from cache.
        """
        if self._anchors_initialized:
            return

        # Try loading from cache first
        if await self._load_anchor_cache():
            self._anchors_initialized = True
            if self._cache:
                self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            return

        # Import INTENT_DATA from keyword_intent
        from .keyword_intent import KeywordIntentCapability
        intent_data = KeywordIntentCapability.INTENT_DATA

        # Get areas from Home Assistant area registry
        from homeassistant.helpers import area_registry
        areas = []
        area_ids_to_names = {}
        registry = area_registry.async_get(self.hass)
        for area in registry.async_list_areas():
            areas.append(area.name)
            area_ids_to_names[area.id] = area.name

        # Get entities grouped by domain and area
        entities_by_domain_area = {}
        try:
            from homeassistant.helpers import entity_registry

            ent_registry = entity_registry.async_get(self.hass)

            for entity in ent_registry.entities.values():
                if entity.disabled:
                    continue
                domain = entity.entity_id.split(".")[0]
                if domain not in intent_data:
                    continue

                area_name = None
                if entity.area_id:
                    area_name = area_ids_to_names.get(entity.area_id)

                if not area_name:
                    continue

                friendly_name = entity.name or entity.original_name
                if not friendly_name:
                    continue

                if domain not in entities_by_domain_area:
                    entities_by_domain_area[domain] = {}
                if area_name not in entities_by_domain_area[domain]:
                    entities_by_domain_area[domain][area_name] = []

                entities_by_domain_area[domain][area_name].append(
                    (entity.entity_id, friendly_name)
                )
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Could not get entities: %s", e)

        total_entities = sum(
            len(entities)
            for domain_areas in entities_by_domain_area.values()
            for entities in domain_areas.values()
        )
        _LOGGER.info(
            "[SemanticCache] Generating anchors for %d areas, %d domains, %d entities (one-time, will cache)",
            len(areas),
            len(intent_data),
            total_entities,
        )

        new_anchors = []
        
        # Track processed area+domain+intent combinations for area-scope (avoid duplicates)
        processed_area_domain_intent = set()
        
        _LOGGER.info("[SemanticCache] Generating anchors...")

        # Generate AREA-SCOPE and ENTITY-SCOPE anchors
        if entities_by_domain_area:
            for domain, areas_entities in entities_by_domain_area.items():
                device_word = DOMAIN_DEVICE_WORDS.get(domain, f"das {domain}")
                
                # Get domain-specific patterns
                area_patterns = AREA_PHRASE_PATTERNS.get(domain, [])
                entity_patterns = ENTITY_PHRASE_PATTERNS.get(domain, [])

                for area_name, entity_list in areas_entities.items():
                    if not entity_list:
                        continue

                    # --- TIER 1: AREA-SCOPE ---
                    # Create ONE entry per area+domain+intent
                    for pattern_tuple in area_patterns:
                        pattern, intent, extra_slots = pattern_tuple
                        
                        area_key = (domain, area_name, intent)
                        if area_key in processed_area_domain_intent:
                            continue
                        processed_area_domain_intent.add(area_key)
                        
                        try:
                            text = pattern.format(area=area_name, device=device_word)
                        except KeyError:
                            continue

                        if len(text.split()) < MIN_CACHE_WORDS:
                            _LOGGER.debug("[SemanticCache] Text too short to cache: '%s'", text)
                            return

                        # Normalize text for Generalized Number Matching
                        # We store the normalized version (e.g. "50 Prozent") so it matches future queries
                        text_norm, _ = self._normalize_numeric_value(text)
                        if text_norm != text:
                            _LOGGER.debug("[SemanticCache] Generalized Storage: Storing '%s' as '%s'", text, text_norm)
                            text = text_norm

                        embedding = await self._get_embedding(text)
                        if embedding is None:
                            continue

                        slots = {"area": area_name, "domain": domain, **extra_slots}
                        
                        # Use all entities in the area
                        area_entity_ids = [e[0] for e in entity_list]

                        entry = CacheEntry(
                            text=text,
                            embedding=embedding.tolist(),
                            intent=intent,
                            entity_ids=area_entity_ids,
                            slots=slots,
                            required_disambiguation=(len(area_entity_ids) > 1),
                            disambiguation_options=None,
                            hits=0,
                            last_hit="",
                            verified=True,
                            generated=True,
                        )
                        new_anchors.append(entry)

                    # --- TIER 2: ENTITY-SCOPE ---
                    # Create ONE entry per entity+intent
                    for entity_id, entity_name in entity_list:
                        for pattern_tuple in entity_patterns:
                            pattern, intent, extra_slots = pattern_tuple
                            
                            try:
                                text = pattern.format(
                                    area=area_name,
                                    device=device_word,
                                    entity_name=entity_name,
                                )
                            except KeyError:
                                continue

                            embedding = await self._get_embedding(text)
                            if embedding is None:
                                continue

                            slots = {"area": area_name, "domain": domain, "name": entity_name, **extra_slots}
                            entry = CacheEntry(
                                text=text,
                                embedding=embedding.tolist(),
                                intent=intent,
                                entity_ids=[entity_id],  # Single entity!
                                slots=slots,
                                required_disambiguation=False,
                                disambiguation_options=None,
                                hits=0,
                                last_hit="",
                                verified=True,
                                generated=True,
                            )
                            new_anchors.append(entry)

                    _LOGGER.info(
                        "[SemanticCache] ✓ %s/%s done - %d entries so far",
                        domain, area_name, len(new_anchors)
                    )

        # Generate global anchors (no area, domain-wide)
        _LOGGER.info("[SemanticCache] Generating global anchors...")
        for domain, patterns in GLOBAL_PHRASE_PATTERNS.items():
            for text, intent, extra_slots in patterns:
                embedding = await self._get_embedding(text)
                if embedding is None:
                    continue
                
                slots = {"domain": domain}
                slots.update(extra_slots)
                
                entry = CacheEntry(
                    text=text,
                    embedding=embedding.tolist(),
                    intent=intent,
                    entity_ids=[],  # Empty - full domain resolution
                    slots=slots,
                    required_disambiguation=False,
                    disambiguation_options=None,
                    hits=0,
                    last_hit="",
                    verified=True,
                    generated=True,
                )
                new_anchors.append(entry)
        
        _LOGGER.info("[SemanticCache] Added %d global anchors", len(GLOBAL_PHRASE_PATTERNS.get("light", [])) + len(GLOBAL_PHRASE_PATTERNS.get("cover", [])))

        # Add to cache and save
        self._cache.extend(new_anchors)
        await self._save_anchor_cache(new_anchors)

        # Rebuild embeddings matrix
        if self._cache:
            self._embeddings_matrix = np.array([e.embedding for e in self._cache])

        self._anchors_initialized = True
        _LOGGER.info("[SemanticCache] Created and cached %d semantic anchors", len(new_anchors))

    def _normalize_numeric_value(self, text: str) -> Tuple[str, List[Any]]:
        """
        Normalize numeric values in text for generalized cache lookup.
        Returns: (normalized_text, extracted_values)
        
        Example: "Setze Rollo auf 75%" -> ("Setze Rollo auf 50 Prozent", [75])
        """
        import re
        extracted = []
        
        # Helper to replace and capture
        def replace_percent(match):
            val = match.group(1)
            extracted.append(int(val))
            return "50 Prozent"

        def replace_temp(match):
            val = match.group(1)
            # Handle float or int
            try:
                if "." in val or "," in val:
                    extracted.append(float(val.replace(",", ".")))
                else:
                    extracted.append(int(val))
            except ValueError:
                pass
            return "21 Grad"
            
        # 1. Percentages: "75%", "75 %", "75 Prozent"
        text_norm = re.sub(r"(\d+)\s*(?:%|Prozent|prozent)", replace_percent, text, flags=re.IGNORECASE)
        
        # 2. Temperatures: "23.5 Grad", "23°", "23 Grad"
        if text_norm == text: # Only apply if percent didn't match (simplification)
            text_norm = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:Grad|°|grad)", replace_temp, text_norm)
            
        return text_norm, extracted

    async def _save_cache(self):
        """Persist cache to disk (only user-learned entries, not pre-generated)."""
        # Filter out pre-generated entries - they have their own cache (anchors.json)
        user_entries = [e for e in self._cache if not e.generated]

        data = {
            "version": 4,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
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
        """Compute cosine similarity between query and all cached embeddings."""
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-10)
        return np.dot(matrix_norm, query_norm)

    async def _rerank_candidates(
        self, query: str, candidates: List[Tuple[float, int, CacheEntry]]
    ) -> Optional[Dict[str, Any]]:
        """
        Rerank candidates using reranker API.

        Args:
            query: User query text
            candidates: List of (vector_score, index, entry) tuples

        Returns:
            Best match dict with reranker score, or None if all below threshold
        """
        if not self.reranker_enabled:
            return None

        # Determine mode: auto tries local first
        mode = self.reranker_mode
        if mode == "auto" and self._reranker_mode_resolved is None:
            # Try to load local model
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model, max_length=512)
                self._reranker_mode_resolved = "local"
                _LOGGER.info(
                    "[SemanticCache] Using local reranker: %s", self.reranker_model
                )
            except ImportError:
                self._reranker_mode_resolved = "api"
                _LOGGER.info(
                    "[SemanticCache] sentence-transformers not available, using API mode"
                )
        elif mode in ("local", "api"):
            self._reranker_mode_resolved = mode

        effective_mode = self._reranker_mode_resolved or mode

        try:
            if effective_mode == "local":
                return await self._rerank_local(query, candidates)
            else:
                return await self._rerank_api(query, candidates)
        except Exception as e:
            _LOGGER.error("[SemanticCache] Reranker failed: %s", e)
            return None

    async def _rerank_local(
        self, query: str, candidates: List[Tuple[float, int, CacheEntry]]
    ) -> Optional[Dict[str, Any]]:
        """Rerank using local sentence-transformers model."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model, max_length=512)
            except ImportError:
                _LOGGER.error("[SemanticCache] sentence-transformers not installed")
                return None

        pairs = [[query, c[2].text] for c in candidates]

        # Run prediction (sync, but fast)
        scores = self._reranker.predict(pairs)
        probs = 1 / (1 + np.exp(-scores))  # Sigmoid

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])

        _LOGGER.debug(
            "[SemanticCache] Local reranker scores: %s (best: %.4f)",
            [f"{p:.3f}" for p in probs],
            best_prob,
        )

        return self._process_rerank_result(candidates, best_idx, best_prob)

    async def _rerank_api(
        self, query: str, candidates: List[Tuple[float, int, CacheEntry]]
    ) -> Optional[Dict[str, Any]]:
        """Rerank using HTTP API."""
        import aiohttp

        payload = {
            "query": query,
            "candidates": [c[2].text for c in candidates],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._reranker_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    _LOGGER.warning(
                        "[SemanticCache] Reranker API error: %s", await response.text()
                    )
                    return None

                data = await response.json()
                scores = data.get("scores", [])

                if not scores:
                    _LOGGER.warning("[SemanticCache] No scores from reranker API")
                    return None

        best_idx = int(data.get("best_index", 0))
        best_prob = float(data.get("best_score", 0))

        _LOGGER.debug(
            "[SemanticCache] API reranker scores: %s (best: %.4f)",
            [f"{p:.3f}" for p in scores],
            best_prob,
        )
        # Log reranker results with intent for each candidate
        for i, (score, (_, _, entry)) in enumerate(zip(scores, candidates)):
            marker = "→ BEST" if i == best_idx else ""
            _LOGGER.debug(
                "[SemanticCache] Reranked %d: score=%.3f, intent=%s, text='%s' %s",
                i + 1, score, entry.intent, entry.text[:40], marker
            )

        return self._process_rerank_result(candidates, best_idx, best_prob)

    def _process_rerank_result(
        self,
        candidates: List[Tuple[float, int, CacheEntry]],
        best_idx: int,
        best_prob: float,
    ) -> Optional[Dict[str, Any]]:
        """Process reranker result and handle anchors."""
        _, cache_idx, entry = candidates[best_idx]
        
        # Get domain-specific threshold
        domain = None
        if entry.entity_ids:
            # Entity IDs are formatted as "domain.entity_name"
            domain = entry.entity_ids[0].split(".")[0] if "." in entry.entity_ids[0] else None
        threshold = DOMAIN_THRESHOLDS.get(domain, self.reranker_threshold)
        
        if best_prob >= threshold:
            return self._make_result(entry, best_prob, cache_idx, is_reranked=True)

        self._stats["reranker_blocks"] += 1
        _LOGGER.debug(
            "[SemanticCache] Reranker BLOCKED: best score %.4f < threshold %.4f (domain=%s)",
            best_prob,
            threshold,
            domain,
        )
        return None

    def _make_result(
        self, entry: CacheEntry, score: float, cache_idx: int, is_reranked: bool = False
    ) -> Dict[str, Any]:
        """Create result dict from cache entry."""
        entry.hits += 1
        entry.last_hit = time.strftime("%Y-%m-%dT%H:%M:%S")

        return {
            "intent": entry.intent,
            "entity_ids": entry.entity_ids,
            "slots": entry.slots,
            "score": score,
            "required_disambiguation": entry.required_disambiguation,
            "disambiguation_options": entry.disambiguation_options,
            "original_text": entry.text,
            "reranked": is_reranked,
        }

    async def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Two-stage cache lookup.

        Stage 1: Fast vector search to find top-k candidates
        Stage 2: Precise reranking to validate semantic match

        Returns:
            Dict with {intent, entity_ids, slots, score, ...} or None
        """
        if not self.enabled:
            return None

        # Strict Requirement: If reranker is disabled, cache is unreliable (no vector-only fallback)
        if not self.reranker_enabled:
            return None

        # Ensure cache is loaded (fast, from disk)
        await self._load_cache()
        
        # If still initializing in background, cache miss gracefully
        if not self._anchors_initialized or not self._cache or self._embeddings_matrix is None:
            self._stats["cache_misses"] += 1
            return None

        # Pre-check: Skip cache for duration patterns (HassTemporaryControl)
        # These need LLM to extract the exact duration value
        duration_patterns = [
            r"\bfür\s+\d+\s*(minuten?|stunden?|sekunden?)\b",
            r"\bfür\s+(eine?|kurze?)\s*(zeit|weile)\b",
            r"\btemporär\b",
            r"\bvorübergehend\b",
            r"\bzeitlich\s+begrenzt\b",
        ]
        # Implicit brightness/temperature patterns - need clarification to determine direction
        # "zu dunkel" → step_up, "zu hell" → step_down (can't determine from cached slot)
        implicit_patterns = [
            r"\bzu\s+dunkel\b",
            r"\bzu\s+hell\b",
            r"\bzu\s+kalt\b",
            r"\bzu\s+warm\b",
            r"\bzu\s+heiß\b",
            r"\bes\s+ist\s+(dunkel|hell)\b",
            r"\b(dunkel|hell)\s+hier\b",
        ]
        import re
        text_lower = text.lower()
        for pattern in duration_patterns + implicit_patterns:
            if re.search(pattern, text_lower):
                _LOGGER.debug(
                    "[SemanticCache] Bypass pattern detected, skipping cache: %s",
                    text[:50]
                )
                return None

        self._stats["total_lookups"] += 1

        # Normalize query for Generalized Number Matching
        query_norm, query_values = self._normalize_numeric_value(text)
        if query_norm != text:
             _LOGGER.debug("[SemanticCache] Generalized Lookup: '%s' -> '%s' [%s]", text, query_norm, query_values)
        
        # Stage 1: Vector search
        _LOGGER.error("DEBUG: Starting vector search for '%s' (norm: '%s')", text, query_norm)
        query_emb = await self._get_embedding(query_norm)
        if query_emb is None:
            self._stats["cache_misses"] += 1
            return None

        similarities = self._cosine_similarity(query_emb, self._embeddings_matrix)
        _LOGGER.error("DEBUG: Similarities calculated. Max: %s", np.max(similarities))

        # Get top-k candidates above loose threshold
        candidates: List[Tuple[float, int, CacheEntry]] = []
        for idx, score in enumerate(similarities):
            if score >= self.vector_threshold:
                candidates.append((float(score), idx, self._cache[idx]))

        _LOGGER.error("DEBUG: Found %d candidates above threshold %s", len(candidates), self.vector_threshold)

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[: self.vector_top_k]

        if not candidates:
            _LOGGER.debug(
                "[SemanticCache] MISS: no candidates above threshold %.2f",
                self.vector_threshold,
            )
            self._stats["cache_misses"] += 1
            return None

        _LOGGER.debug(
            "[SemanticCache] Vector search found %d candidates (top: %.3f)",
            len(candidates),
            candidates[0][0],
        )

        # Stage 2: Reranking
        best_result = await self._rerank_candidates(query_norm, candidates)

        if best_result:
            self._stats["cache_hits"] += 1
            
            # generalized number injection:
            # If we extracted values from query, overwrite the cached slot values
            if query_values:
                val = query_values[0] # Assume primary value
                slots = best_result.get("slots", {})
                
                # Update known numeric slots
                updated = False
                for key in ["position", "brightness", "temperature", "volume_level", "humidity"]:
                    if key in slots:
                        _LOGGER.debug("[SemanticCache] Injecting dynamic value %s into slot '%s'", val, key)
                        slots[key] = val
                        updated = True
                
                if updated:
                    best_result["slots"] = slots

            _LOGGER.info(
                "[SemanticCache] HIT (score=%.3f, reranked=%s): '%s' -> '%s' [%s]",
                best_result["score"],
                best_result.get("reranked", False),
                text[:40],
                best_result["intent"],
                best_result["entity_ids"][0] if best_result["entity_ids"] else "?",
            )
            return best_result

        self._stats["cache_misses"] += 1
        return None

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
        """
        Cache a successful command resolution.

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
        ):
            _LOGGER.debug("[SemanticCache] SKIP: non-repeatable intent %s", intent)
            return

        # Normalize text for Generalized Number Matching
        # We store the normalized version (e.g. "50 Prozent") so it matches future queries
        text_norm, _ = self._normalize_numeric_value(text)
        if text_norm != text:
             _LOGGER.debug("[SemanticCache] Generalized Storage: Storing '%s' as '%s'", text, text_norm)
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

        # Filter out runtime-computed values that shouldn't be cached
        # brightness is calculated at execution time for step_up/step_down
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
            removed = self._cache[self.max_entries :]
            self._cache = self._cache[: self.max_entries]
            self._embeddings_matrix = np.array([e.embedding for e in self._cache])
            _LOGGER.debug("[SemanticCache] Evicted %d old entries", len(removed))

        await self._save_cache()

        _LOGGER.info(
            "[SemanticCache] Stored: '%s' -> %s [%s]",
            text[:40],
            intent,
            entity_ids[0] if entity_ids else "?",
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        await self._load_cache()
        anchor_count = sum(1 for e in self._cache if e.generated)
        real_count = len(self._cache) - anchor_count
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "anchor_count": anchor_count,
            "real_entries": real_count,
            "hit_rate": (
                self._stats["cache_hits"] / self._stats["total_lookups"] * 100
                if self._stats["total_lookups"] > 0
                else 0
            ),
            "embedding_model": self.embedding_model,
            "reranker_host": f"{self.reranker_ip}:{self.reranker_port}",
            "reranker_enabled": self.reranker_enabled,
            "embedding_host": f"{self.embedding_ip}:{self.embedding_port}",
        }

    async def clear(self):
        """Clear all cached entries (including anchors - they'll be regenerated)."""
        self._cache = []
        self._embeddings_matrix = None
        self._anchors_initialized = False
        self._stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "reranker_blocks": 0,
            "anchor_escalations": 0,
        }
        await self._save_cache()
        _LOGGER.info("[SemanticCache] Cache cleared")
