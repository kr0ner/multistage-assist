"""Semantic Cache Builder - Anchor Generation and Cache Creation.

This module is responsible for generating semantic anchor entries for the cache.
Anchors provide pre-verified command patterns that enable fast cache hits without
needing LLM processing.

The SemanticCacheCapability imports this builder for initial cache population.

================================================================================
CRITICAL DESIGN PRINCIPLES - DO NOT VIOLATE
================================================================================

1. INTENT SEPARATION IS PARAMOUNT
   - TurnOn shall NEVER be mistaken for TurnOff (and vice versa)
   - Intent confusion is a critical failure

2. NO MATCH IS ACCEPTABLE
   - If we cannot find a confident match, return None
   - Multiple equal-ranked matches should escalate, not guess

3. WRONG ACTION IS UNACCEPTABLE
   - Doing the wrong thing is a HUGE NO-GO
   - A false positive is worse than a false negative

4. ESCALATE RATHER THAN GUESS
   - When uncertain, escalate to Stage 2 LLM
   - Never execute a command we're not confident about

5. ONLY EXPOSED ENTITIES
   - Only entities exposed to Assist/Conversation should generate anchors
   - async_should_expose must pass for every entity

These principles prioritize PRECISION over RECALL.

================================================================================
ANCHOR GENERATION LOGIC - DO NOT MODIFY WITHOUT UNDERSTANDING THIS!
================================================================================

The anchor generation uses a 4-tier structure. All pattern variants are generated
for comprehensive semantic matching coverage.

+--------+--------------------------------+---------------+----------------------------+
| Tier   | Source                         | Scope         | Count Formula              |
+--------+--------------------------------+---------------+----------------------------+
| AREA   | AREA_PHRASE_PATTERNS[domain]   | Each area     | areas × patterns_per_domain |
| ENTITY | ENTITY_PHRASE_PATTERNS[domain] | Each entity   | entities × patterns_per_domain |
| FLOOR  | Reuses AREA_PHRASE_PATTERNS    | Each floor    | floors × patterns_per_domain |
| GLOBAL | GLOBAL_PHRASE_PATTERNS[domain] | Domain-wide   | patterns                   |
+--------+--------------------------------+---------------+----------------------------+

Key rules:
- ALL pattern variants are generated (not just 1 per intent)
- Deduplication is by TEXT, not by intent (allows multiple phrasings per action)
- Expected total: ~1500-2000 entries for a typical home

Example generation:
    Light domain (~30 patterns):
        - 21 areas × 30 patterns = 630 area anchors
        - ~42 entities × 12 patterns = 504 entity anchors
        - 4 floors × 30 patterns = 120 floor anchors
    + Global patterns (~18 total)
    = ~1500+ total anchors
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from .semantic_cache_types import (
    CacheEntry,
    MIN_CACHE_WORDS,
)

# Increment to force cache regeneration when patterns change
CACHE_VERSION = 6

from .german_utils import get_prepositional_area

# Import async_should_expose at module level for testability
# Falls back to a function that always returns True if HA components not available
try:
    from homeassistant.components.homeassistant.exposed_entities import async_should_expose
except ImportError:
    def async_should_expose(hass, domain, entity_id):
        """Fallback: always return True if exposure check unavailable."""
        return True

_LOGGER = logging.getLogger(__name__)


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




# Import patterns from cache_patterns package
from .cache_patterns.base import (
    DOMAIN_DEVICE_WORDS,
    DOMAIN_DEVICE_WORDS_SINGULAR,
    DOMAIN_DEVICE_WORDS_NOMINATIVE,
    DOMAIN_DEVICE_WORDS_DATIVE
)
from .cache_patterns.area import AREA_PHRASE_PATTERNS
from .cache_patterns.entity import ENTITY_PHRASE_PATTERNS
from .cache_patterns.global_patterns import GLOBAL_PHRASE_PATTERNS

# UNIQUE-ENTITY-SCOPE patterns: {device} + {entity_name} → single entity (NO AREA)
# For entities with GLOBALLY UNIQUE names (e.g. "Ambilight", "Weihnachtsbaum")
# Must not clash with Area/Floor names.




class SemanticCacheBuilder:
    """Builds semantic anchor cache entries.
    
    This class handles:
    - Loading existing anchor cache from disk
    - Generating new anchors based on areas, entities, and patterns
    - Saving anchors to disk for fast subsequent startups
    """

    def __init__(
        self, 
        hass, 
        config: Dict[str, Any],
        get_embedding_func,
        normalize_func,
        batch_embedding_func=None
    ):
        """Initialize builder."""
        self.hass = hass
        self.config = config
        self._get_embedding = get_embedding_func
        self._batch_embed = batch_embedding_func
        self._normalize_numeric_value = normalize_func

    async def load_anchor_cache(self) -> Tuple[bool, List[CacheEntry]]:
        """Load anchor cache from disk.
        
        Returns:
            Tuple of (success, entries)
        """
        anchor_file = os.path.join(
            self.hass.config.path(".storage"), "multistage_assist_anchors.json"
        )
        if not os.path.exists(anchor_file):
            return False, []

        try:
            def _read():
                with open(anchor_file, "r") as f:
                    return json.load(f)

            data = await self.hass.async_add_executor_job(_read)
            
            # Version check - force regeneration if version mismatch
            if data.get("version") != CACHE_VERSION:
                _LOGGER.info(
                    "[SemanticCache] Cache version mismatch (found %s, need %s). Regenerating.",
                    data.get("version"),
                    CACHE_VERSION
                )
                return False, []
            
            # Load anchors (add-on handles model consistency)
            entries = []
            for entry_data in data.get("anchors", []):
                # Sanitize removed fields
                entry_data.pop("is_anchor", None)
                entries.append(CacheEntry(**entry_data))
            
            _LOGGER.info("[SemanticCache] Loaded %d anchors from cache", len(entries))
            return True, entries
        except Exception as e:
            _LOGGER.warning("[SemanticCache] Failed to load anchor cache: %s", e)
            return False, []

    async def save_anchor_cache(self, anchors: List[CacheEntry]):
        """Save anchor entries to separate cache file."""
        anchor_file = os.path.join(
            self.hass.config.path(".storage"), "multistage_assist_anchors.json"
        )
        
        data = {
            "version": CACHE_VERSION,
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

    async def _create_anchor_entry(
        self,
        text: str,
        intent: str,
        slots: Dict[str, Any],
        entity_ids: List[str] = None,
        required_disambiguation: bool = False,
        generated: bool = True
    ) -> Optional[CacheEntry]:
        """Create a cache entry with validation and embedding.
        
        Handles:
        1. Minimum word count check
        2. Numeric normalization
        3. Embedding generation
        4. CacheEntry instantiation
        """
        if len(text.split()) < MIN_CACHE_WORDS:
            return None

        # Normalize text for Generalized Number Matching
        text_norm, _ = self._normalize_numeric_value(text)
        if text_norm != text:
            text = text_norm

        _LOGGER.debug("[SemanticCache] Creating entry for: '%s'", text)
        
        # Embedding can be deferred for batching
        emb_list = None
        if not self._batch_embed:
            embedding = await self._get_embedding(text)
            if embedding is None:
                _LOGGER.warning("[SemanticCache] Failed to generate embedding for anchor: '%s'", text)
                return None
            emb_list = embedding.tolist()

        return CacheEntry(
            text=text,
            embedding=emb_list,
            intent=intent,
            entity_ids=entity_ids or [],
            slots=slots,
            required_disambiguation=required_disambiguation,
            disambiguation_options=None,
            hits=0,
            last_hit="",
            verified=True,
            generated=generated,
        )

    async def generate_anchors(self) -> List[CacheEntry]:
        """Generate semantic anchor entries for each domain × intent × area × entity.
        
        Returns:
            List of generated CacheEntry objects
        """
        # Import INTENT_DATA from keyword_intent
        from ..capabilities.keyword_intent import KeywordIntentCapability
        intent_data = KeywordIntentCapability.INTENT_DATA

        # Get areas from Home Assistant area registry
        from homeassistant.helpers import area_registry, floor_registry
        areas = []
        area_ids_to_names = {}
        registry = area_registry.async_get(self.hass)
        for area in registry.async_list_areas():
            areas.append(area.name)
            area_ids_to_names[area.id] = area.name

        # Get floors from Home Assistant floor registry
        floors = []
        floor_ids_to_names = {}
        floor_reg = floor_registry.async_get(self.hass)
        for floor in floor_reg.async_list_floors():
            floors.append(floor.name)
            floor_ids_to_names[floor.floor_id] = floor.name

        # Map area_id -> floor_name
        area_id_to_floor = {}
        for area in registry.async_list_areas():
            if area.floor_id and area.floor_id in floor_ids_to_names:
                area_id_to_floor[area.id] = floor_ids_to_names[area.floor_id]

        # Get entities grouped by domain and area
        entities_by_domain_area = {}
        entities_by_domain_floor = {}  # NEW: entities by floor
        
        # Track global name usage for unique entity anchors
        from collections import defaultdict
        global_name_counts = defaultdict(int)
        entities_by_name = defaultdict(list) # name -> list of (domain, entity_id)
        try:
            from homeassistant.helpers import entity_registry

            ent_registry = entity_registry.async_get(self.hass)

            for entity in ent_registry.entities.values():
                if entity.disabled:
                    continue
                domain = entity.entity_id.split(".")[0]
                if domain not in intent_data:
                    continue

                # Check exposure to Assist/Conversation
                try:
                    exposed = async_should_expose(self.hass, "conversation", entity.entity_id)
                    if not exposed:
                        _LOGGER.debug(f"[SemanticCache] Skipping {entity.entity_id} (Not exposed)")
                        continue
                except Exception as e:
                    # Fallback: If we can't verify exposure, INCLUDE IT (Fail Open for old HA versions)
                    _LOGGER.debug(f"[SemanticCache] Exposure check unavailable for {entity.entity_id}: {e}. Including.")

                area_name = None
                floor_name = None
                if entity.area_id:
                    area_name = area_ids_to_names.get(entity.area_id)
                    floor_name = area_id_to_floor.get(entity.area_id)

                friendly_name = entity.name or entity.original_name
                if not friendly_name:
                    continue

                # Track for global uniqueness
                clean_name = friendly_name.strip()
                global_name_counts[clean_name] += 1
                entities_by_name[clean_name].append((domain, entity.entity_id))

                # Add to area dict
                if area_name:
                    if domain not in entities_by_domain_area:
                        entities_by_domain_area[domain] = {}
                    if area_name not in entities_by_domain_area[domain]:
                        entities_by_domain_area[domain][area_name] = []
                    entities_by_domain_area[domain][area_name].append(
                        (entity.entity_id, friendly_name)
                    )

                # Add to floor dict
                if floor_name:
                    if domain not in entities_by_domain_floor:
                        entities_by_domain_floor[domain] = {}
                    if floor_name not in entities_by_domain_floor[domain]:
                        entities_by_domain_floor[domain][floor_name] = []
                    entities_by_domain_floor[domain][floor_name].append(
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
            "[SemanticCache] Generating anchors for %d areas, %d floors, %d domains, %d entities",
            len(areas),
            len(floors),
            len(intent_data),
            total_entities,
        )

        new_anchors = []

        # Generate UNIQUE ENTITY anchors (Global)
        _LOGGER.info("[SemanticCache] Generating unique entity anchors...")
        forbidden_names = set(a.lower() for a in areas) | set(f.lower() for f in floors)
        
        new_anchors.extend(
            await self._generate_unique_entity_anchors(
                global_name_counts, entities_by_name, forbidden_names
            )
        )
        
        
        # Track processed area+domain+intent combinations for area-scope (avoid duplicates)
        processed_area_domain_intent = set()
        processed_floor_domain_intent = set()
        
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
                    # Use singular device word if only one entity
                    area_device_word = device_word
                    if len(entity_list) == 1:
                        area_device_word = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, device_word)
                    
                    new_anchors.extend(
                        await self._generate_area_anchors(
                            domain, area_name, entity_list, area_device_word,
                            area_patterns, processed_area_domain_intent
                        )
                    )

                    # --- TIER 2: ENTITY-SCOPE (use singular device word) ---
                    device_word_singular = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, device_word)
                    new_anchors.extend(
                        await self._generate_entity_anchors(
                            domain, area_name, entity_list, device_word_singular, entity_patterns
                        )
                    )

                    _LOGGER.info(
                        "[SemanticCache] ✓ %s/%s done - %d entries so far",
                        domain, area_name, len(new_anchors)
                    )

        # Generate FLOOR-SCOPE anchors (reuse area patterns with floor substitution)
        if entities_by_domain_floor:
            _LOGGER.info("[SemanticCache] Generating floor anchors...")
            for domain, floors_entities in entities_by_domain_floor.items():
                device_word = DOMAIN_DEVICE_WORDS.get(domain, f"das {domain}")
                # Reuse area patterns for floors
                area_patterns = AREA_PHRASE_PATTERNS.get(domain, [])

                for floor_name, entity_list in floors_entities.items():
                    if not entity_list:
                        continue

                    # Use singular device word if only one entity
                    floor_device_word = device_word
                    if len(entity_list) == 1:
                        floor_device_word = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, device_word)

                    new_anchors.extend(
                        await self._generate_floor_anchors(
                            domain, floor_name, entity_list, floor_device_word,
                            area_patterns, processed_floor_domain_intent
                        )
                    )

                    _LOGGER.info(
                        "[SemanticCache] ✓ %s/%s (floor) done - %d entries so far",
                        domain, floor_name, len(new_anchors)
                    )

        # Generate global anchors (no area, domain-wide)
        # CRITICAL: DO NOT introduce local embedding generation here! 
        # All embeddings must come from the add-on via self._batch_embed or self._get_embedding.
        _LOGGER.info("[SemanticCache] Generating global anchors...")
        new_anchors.extend(await self._generate_global_anchors())

        # --- BATCH EMBEDDING ---
        if self._batch_embed and new_anchors:
            _LOGGER.info("[SemanticCache] Batch embedding %d anchors...", len(new_anchors))
            texts = [a.text for a in new_anchors]
            
            # Divide into chunks to avoid too large requests
            CHUNK_SIZE = 100
            for i in range(0, len(texts), CHUNK_SIZE):
                chunk_texts = texts[i:i + CHUNK_SIZE]
                chunk_embeddings = await self._batch_embed(chunk_texts)
                
                if chunk_embeddings:
                    for j, emb in enumerate(chunk_embeddings):
                        new_anchors[i + j].embedding = emb.tolist()
                else:
                    _LOGGER.error("[SemanticCache] Batch embedding failed for chunk %d", i // CHUNK_SIZE)

            # Filter out entries that failed to embed
            new_anchors = [a for a in new_anchors if a.embedding is not None]

        _LOGGER.info("[SemanticCache] Created %d semantic anchors", len(new_anchors))
        return new_anchors

    async def _generate_area_anchors(
        self,
        domain: str,
        area_name: str,
        entity_list: List[Tuple[str, str]],
        device_word: str,
        area_patterns: List[Tuple[str, str, Dict]],
        processed: set
    ) -> List[CacheEntry]:
        """Generate area-scope anchors."""
        anchors = []
        
        # Pre-calculate prepositional form for the area
        full_prep_phrase = get_prepositional_area(area_name)
        # Robustly extract just the prepositional part using case-insensitive split
        # to avoid "Kinder Badezimmer Kinder Badezimmer" duplication
        parts = re.split(re.escape(area_name), full_prep_phrase, maxsplit=1, flags=re.IGNORECASE)
        area_prep = parts[0].strip() if parts else ""
        
        # Generate for both plural and singular device words
        # Users often say "Licht" even if there are multiple.
        device_plural = DOMAIN_DEVICE_WORDS.get(domain, f"die {domain}")
        device_singular = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, f"das {domain}")
        
        device_variants = [device_plural]
        if device_singular != device_plural:
            device_variants.append(device_singular)
            
        for d_word in device_variants:
            for pattern_tuple in area_patterns:
                pattern, intent, extra_slots = pattern_tuple
                
                # Get all case forms for the current device word variant
                device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, d_word)
                device_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, d_word)
                
                try:
                    text = pattern.format(
                        area=area_name, 
                        area_prep=area_prep,
                        device=d_word, 
                        device_nom=device_nom,
                        device_dat=device_dat
                    )
                except KeyError:
                    continue
                
                # Deduplicate by actual generated text (not by intent)
                text_key = (domain, area_name, text)
                if text_key in processed:
                    continue
                processed.add(text_key)
    
                slots = {"area": area_name, "domain": domain, **extra_slots}
                
                # Use all entities in the area
                area_entity_ids = [e[0] for e in entity_list]
                
                # Filter non-dimmable lights for dimming intents
                if domain == "light" and intent == "HassLightSet":
                    dimmable_ids = []
                    for eid in area_entity_ids:
                        state = self.hass.states.get(eid)
                        if state:
                            modes = state.attributes.get("supported_color_modes", [])
                            if not modes or modes != ["onoff"]:
                                dimmable_ids.append(eid)
                    area_entity_ids = dimmable_ids
                    if not area_entity_ids:
                        continue
    
                entry = await self._create_anchor_entry(
                    text=text,
                    intent=intent,
                    slots=slots,
                    entity_ids=area_entity_ids,
                    required_disambiguation=(len(area_entity_ids) > 1),
                    generated=True
                )
                
                if entry:
                    anchors.append(entry)
        
        return anchors

    async def _generate_entity_anchors(
        self,
        domain: str,
        area_name: str,
        entity_list: List[Tuple[str, str]],
        device_word: str,
        entity_patterns: List[Tuple[str, str, Dict]]
    ) -> List[CacheEntry]:
        """Generate entity-scope anchors."""
        anchors = []
        
        for entity_id, entity_name in entity_list:
            # Skip entity only if its name EXACTLY matches the area name
            # E.g., skip "Küche" when area is "Küche" (covered by area pattern)
            # But keep "Küche Spots" - it's a distinct entity needing its own anchor
            entity_name_lower = entity_name.lower()
            area_name_lower = area_name.lower()
            if entity_name_lower == area_name_lower:
                _LOGGER.debug(
                    "[SemanticCache] Skipping entity '%s' - exact match with area '%s'",
                    entity_name, area_name
                )
                continue
            
            # Check dimmability for lights
            is_dimmable = True
            if domain == "light":
                state = self.hass.states.get(entity_id)
                if state:
                    color_modes = state.attributes.get("supported_color_modes", [])
                    is_dimmable = not color_modes or color_modes != ["onoff"]

            for pattern_tuple in entity_patterns:
                pattern, intent, extra_slots = pattern_tuple
                
                # Skip dimming patterns for non-dimmable lights
                if intent == "HassLightSet" and not is_dimmable:
                    continue
                
                # Get all case forms for device word
                device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, device_word)
                device_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, device_word)
                
                try:
                    text = pattern.format(
                        area=area_name,
                        device=device_word,
                        device_nom=device_nom,
                        device_dat=device_dat,
                        entity_name=entity_name,
                    )
                except KeyError:
                    continue

                slots = {"area": area_name, "domain": domain, "name": entity_name, **extra_slots}
                
                entry = await self._create_anchor_entry(
                    text=text,
                    intent=intent,
                    slots=slots,
                    entity_ids=[entity_id],
                    generated=True
                )
                
                if entry:
                    anchors.append(entry)
        
        return anchors

    async def _generate_floor_anchors(
        self,
        domain: str,
        floor_name: str,
        entity_list: List[Tuple[str, str]],
        device_word: str,
        area_patterns: List[Tuple[str, str, Dict]],  # Reuse area patterns
        processed: set
    ) -> List[CacheEntry]:
        """Generate floor-scope anchors (reuses area patterns with floor substitution)."""
        anchors = []
        
        # Pre-calculate prepositional form for the floor
        full_prep_phrase = get_prepositional_area(floor_name)
        parts = re.split(re.escape(floor_name), full_prep_phrase, maxsplit=1, flags=re.IGNORECASE)
        area_prep = parts[0].strip() if parts else ""
        
        # Generate for both plural and singular device words
        device_plural = DOMAIN_DEVICE_WORDS.get(domain, f"die {domain}")
        device_singular = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, f"das {domain}")
        
        device_variants = [device_plural]
        if device_singular != device_plural:
            device_variants.append(device_singular)
            
        for d_word in device_variants:
            for pattern_tuple in area_patterns:
                pattern, intent, extra_slots = pattern_tuple
                
                # Get all case forms for the current device word variant
                device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, d_word)
                device_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, d_word)
                
                try:
                    # Reuse area patterns - substitute {area} with floor_name and {area_prep} with floor_prep
                    text = pattern.format(
                        area=floor_name, 
                        area_prep=area_prep,
                        device=d_word, 
                        device_nom=device_nom,
                        device_dat=device_dat
                    )
                except KeyError:
                    continue
                
                # Deduplicate by actual generated text (not by intent)
                floor_key = (domain, floor_name, text)
                if floor_key in processed:
                    continue
                processed.add(floor_key)
    
                slots = {"floor": floor_name, "domain": domain, **extra_slots}
                
                # Use all entities on the floor
                floor_entity_ids = [e[0] for e in entity_list]
                
                # Filter non-dimmable lights for dimming intents
                if domain == "light" and intent == "HassLightSet":
                    dimmable_ids = []
                    for eid in floor_entity_ids:
                        state = self.hass.states.get(eid)
                        if state:
                            modes = state.attributes.get("supported_color_modes", [])
                            if not modes or modes != ["onoff"]:
                                dimmable_ids.append(eid)
                    floor_entity_ids = dimmable_ids
                    if not floor_entity_ids:
                        continue
    
                entry = await self._create_anchor_entry(
                    text=text,
                    intent=intent,
                    slots=slots,
                    entity_ids=floor_entity_ids,
                    required_disambiguation=(len(floor_entity_ids) > 1),
                    generated=True
                )
                
                if entry:
                    anchors.append(entry)
        
        return anchors

    async def _generate_global_anchors(self) -> List[CacheEntry]:
        """Generate global (domain-wide) anchors."""
        anchors = []
        
        for domain, patterns in GLOBAL_PHRASE_PATTERNS.items():
            for text, intent, extra_slots in patterns:
                slots = {"domain": domain}
                slots.update(extra_slots)
                
                entry = await self._create_anchor_entry(
                    text=text,
                    intent=intent,
                    slots=slots,
                    entity_ids=[],
                    generated=True
                )
                
                if entry:
                    anchors.append(entry)
        
        return anchors


    async def _generate_unique_entity_anchors(
        self,
        global_name_counts: Dict[str, int],
        entities_by_name: Dict[str, List[Tuple[str, str]]],
        forbidden_names: Set[str]
    ) -> List[CacheEntry]:
        """Generate anchors for globally unique entity names (without area context)."""
        anchors = []
        unique_added = 0
        processed_count = 0

        for name, count in global_name_counts.items():
            if count != 1:
                continue
                
            if name.lower() in forbidden_names:
                continue
            
            processed_count += 1
            if processed_count % 10 == 0:
                 _LOGGER.debug("[SemanticCache] Processing unique entity %d: %s...", processed_count, name)
                
            # Valid unique entity
            domain, entity_id = entities_by_name[name][0]
            
            # Fetch device word (accusative) and extract article
            # e.g. "das Licht" -> "das"
            device_word_acc = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, "das Gerät")
            article_acc = device_word_acc.split()[0] if " " in device_word_acc else ""
            
            # Nominative (for queries)
            device_word_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, "das Gerät")
            # Dative
            device_word_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, "dem Gerät")

            # Domain name to prefix if missing (e.g. "Licht", "Rollladen")
            # Logic: If entity name is "Dusche", prefix "Licht". If "Deckenlicht", prefix nothing.
            domain_noun = device_word_acc.split()[-1] if " " in device_word_acc else ""
            if domain_noun.lower() in name.lower():
                # Already present
                prefix = ""
            else:
                prefix = f" {domain_noun}"

            # Prepare case-specific full device words
            article_acc = device_word_acc.split()[0] if " " in device_word_acc else ""
            full_dev_acc = f"{article_acc}{prefix}".strip()

            article_nom = device_word_nom.split()[0] if " " in device_word_nom else ""
            full_dev_nom = f"{article_nom}{prefix}".strip()

            article_dat = device_word_dat.split()[0] if " " in device_word_dat else ""
            full_dev_dat = f"{article_dat}{prefix}".strip()
            
            # Use ENTITY_PHRASE_PATTERNS but filtered for NO AREA
            entity_patterns = ENTITY_PHRASE_PATTERNS.get(domain, [])
            
            for pattern_template, intent, extra_slots in entity_patterns:
                # SKIP if pattern requires {area}
                if "{area}" in pattern_template:
                    continue
                
                # Determine which device word variant to use based on pattern
                if "{device_nom}" in pattern_template:
                    dev_to_use = full_dev_nom
                elif "{device_dat}" in pattern_template:
                    dev_to_use = full_dev_dat
                else:
                    dev_to_use = full_dev_acc
                
                try:
                    text = pattern_template.format(
                        device=dev_to_use,
                        device_nom=full_dev_nom,
                        device_dat=full_dev_dat,
                        entity_name=name
                    ).replace("  ", " ").strip()
                except KeyError:
                    continue

                slots = extra_slots.copy()
                slots["name"] = name
                
                entry = await self._create_anchor_entry(
                    text=text,
                    intent=intent,
                    slots=slots,
                    entity_ids=[entity_id],
                    generated=True
                )
                
                if entry:
                    anchors.append(entry)
                    unique_added += 1

        _LOGGER.info("[SemanticCache] Added %d unique entity anchors", unique_added)
        return anchors
