"""Semantic Cache Builder - Anchor Generation and Cache Creation.

This module is responsible for generating semantic anchor entries for the cache.
Anchors provide pre-verified command patterns that enable fast cache hits without
needing LLM processing.

The SemanticCacheCapability imports this builder for initial cache population.

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.semantic_cache_types import (
    CacheEntry,
    MIN_CACHE_WORDS,
)

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
AREA_PHRASE_PATTERNS = {
    "light": [
        # === HassTurnOn - multiple word orders ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        # Word order variants (troublesome patterns)
        ("{device} in {area} an", "HassTurnOn", {}),  # "Licht in Küche an"
        ("{device} an in {area}", "HassTurnOn", {}),  # "Licht an in der Küche"
        ("Mach {device} in {area} an", "HassTurnOn", {}),  # "Mach das Licht in Küche an"
        # Colloquial/informal
        ("{area} {device} an", "HassTurnOn", {}),  # "Küche Licht an"
        ("{device} {area} an", "HassTurnOn", {}),  # "Licht Küche an"
        # Synonyms: Lampe, Beleuchtung, einschalten, Aktiviere
        ("die Lampe in {area} an", "HassTurnOn", {}),
        ("Lampe in {area} anschalten", "HassTurnOn", {}),
        ("Mach die Lampe in {area} an", "HassTurnOn", {}),
        ("{area} Lampe an", "HassTurnOn", {}),
        ("Beleuchtung in {area} an", "HassTurnOn", {}),
        ("Aktiviere Beleuchtung in {area}", "HassTurnOn", {}),
        ("Aktiviere {device} in {area}", "HassTurnOn", {}),
        ("{device} in {area} einschalten", "HassTurnOn", {}),
        ("Einschalten {device} in {area}", "HassTurnOn", {}),
        ("Mach mal {device} in {area} an", "HassTurnOn", {}),
        
        # === HassTurnOff - multiple word orders ===
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        # Word order variants
        ("{device} in {area} aus", "HassTurnOff", {}),  # "Licht in Küche aus"
        ("{device} aus in {area}", "HassTurnOff", {}),  # "Licht aus in der Küche"
        ("Mach {device} in {area} aus", "HassTurnOff", {}),  # "Mach das Licht in Küche aus"
        # Colloquial/informal
        ("{area} {device} aus", "HassTurnOff", {}),  # "Küche Licht aus"
        ("{device} {area} aus", "HassTurnOff", {}),  # "Licht Küche aus"
        # Synonyms: Lampe, Beleuchtung, ausschalten, Deaktiviere
        ("die Lampe in {area} aus", "HassTurnOff", {}),
        ("Lampe in {area} ausschalten", "HassTurnOff", {}),
        ("Mach die Lampe in {area} aus", "HassTurnOff", {}),
        ("{area} Lampe aus", "HassTurnOff", {}),
        ("Beleuchtung in {area} aus", "HassTurnOff", {}),
        ("Deaktiviere Beleuchtung in {area}", "HassTurnOff", {}),
        ("Deaktiviere {device} in {area}", "HassTurnOff", {}),
        ("{device} in {area} ausschalten", "HassTurnOff", {}),
        ("Ausschalten {device} in {area}", "HassTurnOff", {}),
        ("Mach mal {device} in {area} aus", "HassTurnOff", {}),
        
        # === HassLightSet - formal brightness patterns ===
        ("Erhöhe die Helligkeit von {device} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} in {area}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # Informal brightness patterns - "heller/dunkler" variations
        ("Mach {device} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("{device} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("{device} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} in {area}", "HassLightSet", {"command": "step_down"}),
        # More informal brightness variants
        ("{device} heller in {area}", "HassLightSet", {"command": "step_up"}),
        ("{device} dunkler in {area}", "HassLightSet", {"command": "step_down"}),
        
        # === HassGetState ===
        ("Ist {device} in {area} an", "HassGetState", {}),
        ("Brennt {device} in {area}", "HassGetState", {}),
        
        # === HassDelayedControl - delayed on/off ===
        ("Schalte {device} in {area} in 10 Minuten an", "HassDelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "HassDelayedControl", {"command": "off"}),
        ("Mach {device} in {area} in 5 Minuten an", "HassDelayedControl", {"command": "on"}),
        ("Mach {device} in {area} um 15 Uhr an", "HassDelayedControl", {"command": "on"}),
        ("Mach {device} in {area} um 15 Uhr aus", "HassDelayedControl", {"command": "off"}),
    ],
    "cover": [
        # === Cover Open ===
        ("Öffne {device} in {area}", "HassSetPosition", {"position": 100}),
        ("{device} in {area} öffnen", "HassSetPosition", {"position": 100}),
        ("{device} in {area} hoch", "HassSetPosition", {"position": 100}),
        ("Mach {device} in {area} auf", "HassSetPosition", {"position": 100}),
        # === Cover Close ===
        ("Schließe {device} in {area}", "HassSetPosition", {"position": 0}),
        ("{device} in {area} schließen", "HassSetPosition", {"position": 0}),
        ("{device} in {area} runter", "HassSetPosition", {"position": 0}),
        ("Mach {device} in {area} zu", "HassSetPosition", {"position": 0}),
        # === Cover Step ===
        ("Fahre {device} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        # === Cover State ===
        ("Ist {device} in {area} offen", "HassGetState", {}),
        ("Sind {device} in {area} offen", "HassGetState", {}),
    ],
    "climate": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Stelle {device} in {area} auf 21 Grad", "HassClimateSetTemperature", {}),
        ("Mach es in {area} wärmer", "HassClimateSetTemperature", {"command": "step_up"}),
        ("Mach es in {area} kälter", "HassClimateSetTemperature", {"command": "step_down"}),
        ("Wie warm ist es in {area}", "HassGetState", {}),
    ],
    "switch": [
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("{device} in {area} an", "HassTurnOn", {}),
        ("Mach {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("{device} in {area} aus", "HassTurnOff", {}),
        ("Mach {device} in {area} aus", "HassTurnOff", {}),
        ("Ist {device} in {area} an", "HassGetState", {}),
        # HassDelayedControl
        ("Schalte {device} in {area} in 10 Minuten an", "HassDelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "HassDelayedControl", {"command": "off"}),
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
        # Formal brightness patterns
        ("Erhöhe die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} {entity_name} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # Informal brightness patterns - "heller/dunkler" variations
        ("Mach {device} {entity_name} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} {entity_name} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("{device} {entity_name} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("{device} {entity_name} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        # State query
        ("Ist {device} {entity_name} in {area} an", "HassGetState", {}),
    ],
    "cover": [
        # Entity patterns use singular device word: "den Rollladen {entity_name}"
        ("Öffne {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Schließe {device} {entity_name} in {area}", "HassTurnOff", {}),
        ("Fahre {device} {entity_name} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} {entity_name} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} {entity_name} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Ist {device} {entity_name} in {area} offen", "HassGetState", {"state": "open"}),
        ("Ist {device} {entity_name} in {area} geschlossen", "HassGetState", {"state": "closed"}),
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

# Import keywords to generate device words with articles
from ..constants.entity_keywords import (
    LIGHT_KEYWORDS,
    COVER_KEYWORDS,
    SWITCH_KEYWORDS,
    FAN_KEYWORDS,
    MEDIA_KEYWORDS,
    SENSOR_KEYWORDS,
    CLIMATE_KEYWORDS,
)

def _get_first_keyword(keywords_dict):
    """Get first keyword (singular form with article)."""
    return next(iter(keywords_dict.keys()))

def _get_first_plural(keywords_dict):
    """Get first keyword's plural form with article."""
    return next(iter(keywords_dict.values()))

# Generic device words by domain (plural form for area/floor scope)
# Auto-generated from entity_keywords with articles
DOMAIN_DEVICE_WORDS = {
    "light": _get_first_keyword(LIGHT_KEYWORDS),        # "das licht"
    "cover": _get_first_plural(COVER_KEYWORDS),         # "die rollläden" (plural for area scope)
    "climate": _get_first_keyword(CLIMATE_KEYWORDS),    # "das thermostat"
    "switch": _get_first_keyword(SWITCH_KEYWORDS),      # "der schalter"
    "fan": _get_first_keyword(FAN_KEYWORDS),            # "der ventilator"
    "media_player": _get_first_keyword(MEDIA_KEYWORDS), # "der tv"
    "sensor": _get_first_keyword(SENSOR_KEYWORDS),      # "der sensor"
    "automation": "die Automatisierung",
}

# Singular device words for entity-scope patterns (single entity)
DOMAIN_DEVICE_WORDS_SINGULAR = {
    "light": _get_first_keyword(LIGHT_KEYWORDS),        # "das licht"
    "cover": _get_first_keyword(COVER_KEYWORDS),        # "der rollladen" (singular!)
    "climate": _get_first_keyword(CLIMATE_KEYWORDS),
    "switch": _get_first_keyword(SWITCH_KEYWORDS),
    "fan": _get_first_keyword(FAN_KEYWORDS),
    "media_player": _get_first_keyword(MEDIA_KEYWORDS),
    "sensor": _get_first_keyword(SENSOR_KEYWORDS),
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


class SemanticCacheBuilder:
    """Builds semantic anchor cache entries.
    
    This class handles:
    - Loading existing anchor cache from disk
    - Generating new anchors based on areas, entities, and patterns
    - Saving anchors to disk for fast subsequent startups
    """

    def __init__(self, hass, config, get_embedding_func, normalize_func):
        """Initialize builder.
        
        Args:
            hass: Home Assistant instance
            config: Configuration dict
            get_embedding_func: Async function to get embeddings
            normalize_func: Function to normalize numeric values in text
        """
        self.hass = hass
        self.config = config
        self._get_embedding = get_embedding_func
        self._normalize_numeric_value = normalize_func
        self.embedding_model = config.get("embedding_model", "bge-m3")

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
            
            # Check if anchor cache is compatible
            if data.get("embedding_model") != self.embedding_model:
                _LOGGER.info("[SemanticCache] Anchor model mismatch, regenerating")
                return False, []
            
            # Load anchors
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

    async def generate_anchors(self) -> List[CacheEntry]:
        """Generate semantic anchor entries for each domain × intent × area × entity.
        
        Returns:
            List of generated CacheEntry objects
        """
        # Import INTENT_DATA from keyword_intent
        from .keyword_intent import KeywordIntentCapability
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
                floor_name = None
                if entity.area_id:
                    area_name = area_ids_to_names.get(entity.area_id)
                    floor_name = area_id_to_floor.get(entity.area_id)

                friendly_name = entity.name or entity.original_name
                if not friendly_name:
                    continue

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
                    new_anchors.extend(
                        await self._generate_area_anchors(
                            domain, area_name, entity_list, device_word,
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

                    new_anchors.extend(
                        await self._generate_floor_anchors(
                            domain, floor_name, entity_list, device_word,
                            area_patterns, processed_floor_domain_intent
                        )
                    )

                    _LOGGER.info(
                        "[SemanticCache] ✓ %s/%s (floor) done - %d entries so far",
                        domain, floor_name, len(new_anchors)
                    )

        # Generate global anchors (no area, domain-wide)
        _LOGGER.info("[SemanticCache] Generating global anchors...")
        new_anchors.extend(await self._generate_global_anchors())

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
        
        for pattern_tuple in area_patterns:
            pattern, intent, extra_slots = pattern_tuple
            
            try:
                text = pattern.format(area=area_name, device=device_word)
            except KeyError:
                continue
            
            # Deduplicate by actual generated text (not by intent)
            text_key = (domain, area_name, text)
            if text_key in processed:
                continue
            processed.add(text_key)

            if len(text.split()) < MIN_CACHE_WORDS:
                continue

            # Normalize text for Generalized Number Matching
            text_norm, _ = self._normalize_numeric_value(text)
            if text_norm != text:
                text = text_norm

            embedding = await self._get_embedding(text)
            if embedding is None:
                continue

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
                    entity_ids=[entity_id],
                    slots=slots,
                    required_disambiguation=False,
                    disambiguation_options=None,
                    hits=0,
                    last_hit="",
                    verified=True,
                    generated=True,
                )
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
        
        for pattern_tuple in area_patterns:
            pattern, intent, extra_slots = pattern_tuple
            
            try:
                # Reuse area patterns - substitute {area} with floor_name
                text = pattern.format(area=floor_name, device=device_word)
            except KeyError:
                continue
            
            # Deduplicate by actual generated text (not by intent)
            floor_key = (domain, floor_name, text)
            if floor_key in processed:
                continue
            processed.add(floor_key)

            if len(text.split()) < MIN_CACHE_WORDS:
                continue

            # Normalize text for Generalized Number Matching
            text_norm, _ = self._normalize_numeric_value(text)
            if text_norm != text:
                text = text_norm

            embedding = await self._get_embedding(text)
            if embedding is None:
                continue

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

            entry = CacheEntry(
                text=text,
                embedding=embedding.tolist(),
                intent=intent,
                entity_ids=floor_entity_ids,
                slots=slots,
                required_disambiguation=(len(floor_entity_ids) > 1),
                disambiguation_options=None,
                hits=0,
                last_hit="",
                verified=True,
                generated=True,
            )
            anchors.append(entry)
        
        return anchors

    async def _generate_global_anchors(self) -> List[CacheEntry]:
        """Generate global (domain-wide) anchors."""
        anchors = []
        
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
                    entity_ids=[],
                    slots=slots,
                    required_disambiguation=False,
                    disambiguation_options=None,
                    hits=0,
                    last_hit="",
                    verified=True,
                    generated=True,
                )
                anchors.append(entry)
        
        return anchors
