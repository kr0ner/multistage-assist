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
from typing import Any, Dict, List, Optional, Tuple, Set

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
        ("Mach mal {device} in {area} an", "HassTurnOn", {}),
        # Colloquial "anmachen"
        ("Kannst du {device} in {area} anmachen", "HassTurnOn", {}),
        ("{device} in {area} anmachen", "HassTurnOn", {}),
        
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
        ("Mach mal {device} in {area} aus", "HassTurnOff", {}),
        
        # === HassLightSet - formal brightness patterns ===
        # Use dative case after 'von': "von dem Licht"
        ("Erhöhe die Helligkeit von {device_dat} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device_dat} in {area}", "HassLightSet", {"command": "step_down"}),
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
        # "Mehr/Weniger" variants
        ("Mehr Licht in {area}", "HassLightSet", {"command": "step_up"}),
        ("Weniger Licht in {area}", "HassLightSet", {"command": "step_down"}),
        ("Mehr Helligkeit in {area}", "HassLightSet", {"command": "step_up"}),
        ("Weniger Helligkeit in {area}", "HassLightSet", {"command": "step_down"}),
        
        # === HassGetState (nominative + question mark) ===
        ("Ist {device_nom} in {area} an?", "HassGetState", {}),
        ("Brennt {device_nom} in {area}?", "HassGetState", {}),
        
        # === DelayedControl - delayed on/off (no specific numbers - delay parsed separately) ===
        # "in X Minuten" / "um X Uhr" = execute AFTER delay
        ("Schalte {device} in {area} in Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in Minuten aus", "DelayedControl", {"command": "off"}),
        ("Mach {device} in {area} in Minuten an", "DelayedControl", {"command": "on"}),
        ("Mach {device} in {area} in Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} in {area} um Uhr an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} um Uhr aus", "DelayedControl", {"command": "off"}),
        ("Mach {device} in {area} um Uhr an", "DelayedControl", {"command": "on"}),
        ("Mach {device} in {area} um Uhr aus", "DelayedControl", {"command": "off"}),
        
        # === TemporaryControl - temporary on/off (no specific numbers - duration parsed separately) ===
        # "für X Minuten" = execute NOW, revert after duration
        ("Schalte {device} in {area} für Minuten an", "TemporaryControl", {"command": "on"}),
        ("Schalte {device} in {area} für Minuten aus", "TemporaryControl", {"command": "off"}),
        ("Mach {device} in {area} für Minuten an", "TemporaryControl", {"command": "on"}),
        ("Mach {device} in {area} für Minuten aus", "TemporaryControl", {"command": "off"}),
        ("{device} in {area} für Minuten an", "TemporaryControl", {"command": "on"}),
        ("{device} in {area} für Minuten aus", "TemporaryControl", {"command": "off"}),
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
        # === Cover Step (formal) ===
        ("Fahre {device} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        # === Cover Step (relative - open more) ===
        ("Öffne {device} in {area} ein bisschen mehr", "HassSetPosition", {"command": "step_up"}),
        ("Öffne {device} in {area} ein wenig", "HassSetPosition", {"command": "step_up"}),
        ("Öffne {device} in {area} etwas mehr", "HassSetPosition", {"command": "step_up"}),
        ("{device} in {area} etwas mehr öffnen", "HassSetPosition", {"command": "step_up"}),
        ("{device} in {area} ein bisschen mehr auf", "HassSetPosition", {"command": "step_up"}),
        ("Mach {device} in {area} etwas weiter auf", "HassSetPosition", {"command": "step_up"}),
        # === Cover Step (relative - close more) ===
        ("Schließe {device} in {area} ein bisschen mehr", "HassSetPosition", {"command": "step_down"}),
        ("Schließe {device} in {area} ein wenig", "HassSetPosition", {"command": "step_down"}),
        ("Schließe {device} in {area} etwas mehr", "HassSetPosition", {"command": "step_down"}),
        ("{device} in {area} etwas mehr schließen", "HassSetPosition", {"command": "step_down"}),
        ("{device} in {area} ein bisschen mehr zu", "HassSetPosition", {"command": "step_down"}),
        ("Mach {device} in {area} etwas weiter zu", "HassSetPosition", {"command": "step_down"}),
        # === Cover State (nominative + question mark) ===
        ("Ist {device_nom} in {area} offen?", "HassGetState", {}),
        ("Sind {device} in {area} offen?", "HassGetState", {}),
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
        # DelayedControl (no specific numbers - delay parsed separately)
        ("Schalte {device} in {area} in Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in Minuten aus", "DelayedControl", {"command": "off"}),
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
# For rooms with multiple entities of same domain, e.g.:
#   "Öffne den Rollladen Büro Nord im Büro" (specific)
#   vs "Öffne die Rollläden im Büro" (all in area)
# NOTE: {device} = accusative case, {device_nom} = nominative case (for questions)
ENTITY_PHRASE_PATTERNS = {
    "light": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        # Formal brightness patterns
        ("Erhöhe die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} {entity_name} in {area}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} {entity_name} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # Informal brightness patterns
        ("Mach {device} {entity_name} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} {entity_name} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("{device} {entity_name} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("{device} {entity_name} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        # State query (nominative case + question mark)
        ("Ist {device_nom} {entity_name} in {area} an?", "HassGetState", {}),
        # UNIQUE (No Area)
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
        ("Mach {device} {entity_name} an", "HassTurnOn", {}),
        ("Mach {device} {entity_name} aus", "HassTurnOff", {}),
        ("{device} {entity_name} an", "HassTurnOn", {}),
        ("{device} {entity_name} aus", "HassTurnOff", {}),
        # Formal brightness (Unique)
        ("Erhöhe die Helligkeit von {device} {entity_name}", "HassLightSet", {"command": "step_up"}),
        ("Reduziere die Helligkeit von {device} {entity_name}", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} {entity_name} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # Informal brightness (Unique)
        ("Mach {device} {entity_name} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} {entity_name} dunkler", "HassLightSet", {"command": "step_down"}),
        ("{device} {entity_name} heller", "HassLightSet", {"command": "step_up"}),
        ("{device} {entity_name} dunkler", "HassLightSet", {"command": "step_down"}),
        # Query (Unique)
        ("Ist {device_nom} {entity_name} an?", "HassGetState", {}),
        ("Ist {device_nom} {entity_name} aus?", "HassGetState", {}),
    ],
    "cover": [
        ("Öffne {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Schließe {device} {entity_name} in {area}", "HassTurnOff", {}),
        ("Fahre {device} {entity_name} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} {entity_name} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} {entity_name} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        # State queries (nominative case + question mark)
        ("Ist {device_nom} {entity_name} in {area} offen?", "HassGetState", {"state": "open"}),
        ("Ist {device_nom} {entity_name} in {area} geschlossen?", "HassGetState", {"state": "closed"}),
        # UNIQUE
        ("Öffne {device} {entity_name}", "HassTurnOn", {}),
        ("Schließe {device} {entity_name}", "HassTurnOff", {}),
        ("Mach {device} {entity_name} auf", "HassTurnOn", {}),
        ("Mach {device} {entity_name} zu", "HassTurnOff", {}),
        ("Fahre {device} {entity_name} hoch", "HassTurnOn", {}),
        ("Fahre {device} {entity_name} runter", "HassTurnOff", {}),
        ("Fahre {device} {entity_name} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} {entity_name} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle {device} {entity_name} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        # Query (Unique)
        ("Ist {device_nom} {entity_name} offen?", "HassGetState", {"state": "open"}),
        ("Ist {device_nom} {entity_name} geschlossen?", "HassGetState", {"state": "closed"}),
    ],
    "climate": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Stelle {device} {entity_name} in {area} auf 21 Grad", "HassClimateSetTemperature", {}),
        # UNIQUE
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
        ("Stelle {device} {entity_name} auf 21 Grad", "HassClimateSetTemperature", {}),
    ],
    "switch": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Ist {device_nom} {entity_name} in {area} an?", "HassGetState", {}),
        # UNIQUE
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
        ("Mach {device} {entity_name} an", "HassTurnOn", {}),
        ("Mach {device} {entity_name} aus", "HassTurnOff", {}),
        ("Ist {device_nom} {entity_name} an?", "HassGetState", {}),
        ("Ist {device_nom} {entity_name} aus?", "HassGetState", {}),
    ],
    "fan": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        # UNIQUE
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
    ],
    "media_player": [
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        # UNIQUE
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
    ],
    "automation": [
        ("Aktiviere {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Deaktiviere {device} {entity_name} in {area}", "HassTurnOff", {}),
        # UNIQUE
        ("Aktiviere {device} {entity_name}", "HassTurnOn", {}),
        ("Deaktiviere {device} {entity_name}", "HassTurnOff", {}),
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

# Import german_utils helpers for article case conversion
from ..utils.german_utils import nominative_to_accusative, nominative_to_dative, capitalize_article_phrase


def _get_device_words(domain: str):
    """Get device words for a domain in all grammatical cases.
    
    Returns tuple of (nominative, accusative, dative, plural) 
    all properly capitalized.
    
    Uses entity_keywords.py as source of truth.
    """
    keyword_maps = {
        "light": LIGHT_KEYWORDS,
        "cover": COVER_KEYWORDS,
        "climate": CLIMATE_KEYWORDS,
        "switch": SWITCH_KEYWORDS,
        "fan": FAN_KEYWORDS,
        "media_player": MEDIA_KEYWORDS,
        "sensor": SENSOR_KEYWORDS,
    }
    
    if domain not in keyword_maps:
        # Fallback for domains without keyword mapping
        fallback = f"das {domain.title()}"
        return (fallback, fallback, fallback, f"die {domain.title()}s")
    
    keywords = keyword_maps[domain]
    singular_nom = _get_first_keyword(keywords)  # e.g. "der rollladen"
    plural = _get_first_plural(keywords)          # e.g. "die rollläden"
    
    # Capitalize properly
    singular_nom = capitalize_article_phrase(singular_nom)  # "der Rollladen"
    plural = capitalize_article_phrase(plural)              # "die Rollläden"
    
    # Derive accusative and dative from nominative
    singular_acc = nominative_to_accusative(singular_nom)   # "den Rollladen"
    singular_dat = nominative_to_dative(singular_nom)       # "dem Rollladen"
    
    return (singular_nom, singular_acc, singular_dat, plural)


# Build device word dictionaries dynamically from entity_keywords
DOMAIN_DEVICE_WORDS = {}           # Plural (for area/floor scope)
DOMAIN_DEVICE_WORDS_SINGULAR = {}  # Singular accusative (for commands)
DOMAIN_DEVICE_WORDS_NOMINATIVE = {} # Singular nominative (for questions)
DOMAIN_DEVICE_WORDS_DATIVE = {}     # Singular dative (for "von dem Licht")

for _domain in ["light", "cover", "climate", "switch", "fan", "media_player", "sensor"]:
    _nom, _acc, _dat, _plural = _get_device_words(_domain)
    DOMAIN_DEVICE_WORDS[_domain] = _plural
    DOMAIN_DEVICE_WORDS_SINGULAR[_domain] = _acc
    DOMAIN_DEVICE_WORDS_NOMINATIVE[_domain] = _nom
    DOMAIN_DEVICE_WORDS_DATIVE[_domain] = _dat

# Add automation (not in entity_keywords)
DOMAIN_DEVICE_WORDS["automation"] = "die Automatisierungen"
DOMAIN_DEVICE_WORDS_SINGULAR["automation"] = "die Automatisierung"
DOMAIN_DEVICE_WORDS_NOMINATIVE["automation"] = "die Automatisierung"
DOMAIN_DEVICE_WORDS_DATIVE["automation"] = "der Automatisierung"

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
        ("Stelle alle Lichter auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # Colloquial Global Match
        ("Alle Lichter an", "HassTurnOn", {}),
        ("Alle Lichter aus", "HassTurnOff", {}),
        # State queries
        ("Welche Lichter sind an?", "HassGetState", {"state": "on"}),
        ("Welche Lichter sind aus?", "HassGetState", {"state": "off"}),
        ("Sind alle Lichter an?", "HassGetState", {"state": "on"}),
        ("Sind alle Lichter aus?", "HassGetState", {"state": "off"}),
    ],
    "cover": [
        ("Schließe alle Rollläden", "HassTurnOff", {}),  # Close = TurnOff
        ("Öffne alle Rollläden", "HassTurnOn", {}),  # Open = TurnOn
        ("Fahre alle Rollläden weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre alle Rollläden weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle alle Rollläden auf 50 Prozent", "HassSetPosition", {"position": 50}),
        # State queries
        ("Welche Rollläden sind offen?", "HassGetState", {"state": "open"}),
        ("Welche Rollläden sind geschlossen?", "HassGetState", {"state": "closed"}),
        ("Welche Rollläden sind zu?", "HassGetState", {"state": "closed"}),
        ("Sind alle Rollläden offen?", "HassGetState", {"state": "open"}),
        ("Sind alle Rollläden geschlossen?", "HassGetState", {"state": "closed"}),
        ("Sind alle Rollläden zu?", "HassGetState", {"state": "closed"}),
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
    # =========================================================================
    # TIMER/CALENDAR - EXCLUDED FROM CACHE (see stage1_cache.py)
    # =========================================================================
    # Timer and calendar commands bypass cache because they often contain
    # variable context that normalization strips out, e.g.:
    #   "Timer auf 15 Minuten der mich an das Gulasch erinnert"
    # The "der mich an das Gulasch erinnert" part is lost during normalization.
    # LLM must handle these to preserve the full command context.
    # =========================================================================
}

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

    def __init__(self, hass, config, get_embedding_func, normalize_func):
        """Initialize builder.
        
        Args:
            hass: Home Assistant instance
            config: Configuration dict
            get_embedding_func: Async function to get embeddings (calls add-on)
            normalize_func: Function to normalize numeric values in text
        """
        self.hass = hass
        self.config = config
        self._get_embedding = get_embedding_func
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
            "version": 2,
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

        _LOGGER.debug("[SemanticCache] Generating embedding for: '%s'", text)
        embedding = await self._get_embedding(text)
        if embedding is None:
            _LOGGER.warning("[SemanticCache] Failed to generate embedding for anchor: '%s'", text)
            return None

        return CacheEntry(
            text=text,
            embedding=embedding.tolist(),
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
            
            # Get all case forms for device word
            device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, device_word)
            device_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, device_word)
            
            try:
                text = pattern.format(
                    area=area_name, 
                    device=device_word, 
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
        
        for pattern_tuple in area_patterns:
            pattern, intent, extra_slots = pattern_tuple
            
            # Get all case forms for device word
            device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, device_word)
            device_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, device_word)
            
            try:
                # Reuse area patterns - substitute {area} with floor_name
                text = pattern.format(
                    area=floor_name, 
                    device=device_word, 
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
            article_nom = device_word_nom.split()[0] if " " in device_word_nom else ""
            # Dative
            device_word_dat = DOMAIN_DEVICE_WORDS_DATIVE.get(domain, "dem Gerät")
            article_dat = device_word_dat.split()[0] if " " in device_word_dat else ""
            
            # Use ENTITY_PHRASE_PATTERNS but filtered for NO AREA
            entity_patterns = ENTITY_PHRASE_PATTERNS.get(domain, [])
            
            for pattern_template, intent, extra_slots in entity_patterns:
                # SKIP if pattern requires {area}
                if "{area}" in pattern_template:
                    continue
                
                # Determine which article to use based on pattern
                if "{device_nom}" in pattern_template:
                    article = article_nom
                elif "{device_dat}" in pattern_template:
                    article = article_dat
                else:
                    article = article_acc
                
                try:
                    text = pattern_template.format(
                        device=article,
                        device_nom=article_nom,
                        device_dat=article_dat,
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
