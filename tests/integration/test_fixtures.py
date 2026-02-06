"""Self-contained test fixtures for semantic cache tests.

This module generates a minimal, synthetic anchor file for testing
that is INDEPENDENT of any user's Home Assistant installation.

The generated anchors cover all core intents and patterns without
requiring a real HA setup.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


# =============================================================================
# SYNTHETIC TEST DATA - Installation Independent
# =============================================================================

# Test areas that will be used in generated anchors
TEST_AREAS = ["Küche", "Büro", "Wohnzimmer", "Schlafzimmer", "Bad"]

# Test floors
TEST_FLOORS = ["Erdgeschoss", "Obergeschoss"]

# Test entity names (with domains)
TEST_ENTITIES = [
    ("light.kuche", "Küche Licht"),
    ("light.buro", "Büro Licht"),
    ("cover.kuche_rollo", "Küche Rollo"),
    ("climate.wohnzimmer", "Wohnzimmer Thermostat"),
]

# Pattern templates for each intent (German)
ANCHOR_PATTERNS = {
    "HassTurnOn": [
        ("Schalte das Licht in {area} an", {"domain": "light"}),
        ("Mach das Licht in {area} an", {"domain": "light"}),
        ("Licht {area} an", {"domain": "light"}),
        ("{area} Licht an", {"domain": "light"}),
        ("Schalte alle Lichter an", {"domain": "light"}),  # Global
        ("Alle Lichter an", {"domain": "light"}),  # Global
    ],
    "HassTurnOff": [
        ("Schalte das Licht in {area} aus", {"domain": "light"}),
        ("Mach das Licht in {area} aus", {"domain": "light"}),
        ("Licht {area} aus", {"domain": "light"}),
        ("{area} Licht aus", {"domain": "light"}),
        ("Schalte alle Lichter aus", {"domain": "light"}),  # Global
        ("Alle Lichter aus", {"domain": "light"}),  # Global
    ],
    "HassLightSet": [
        ("Dimme das Licht in {area}", {"domain": "light", "command": "step_down"}),
        ("Mach das Licht in {area} heller", {"domain": "light", "command": "step_up"}),
        ("Mehr Licht in {area}", {"domain": "light", "command": "step_up"}),
        ("Weniger Licht in {area}", {"domain": "light", "command": "step_down"}),
        ("Setze das Licht in {area} auf 50 Prozent", {"domain": "light", "brightness": 50}),
        ("Alle Lichter auf 50 Prozent", {"domain": "light", "brightness": 50}),  # Global
    ],
    "HassGetState": [
        ("Ist das Licht in {area} an", {"domain": "light"}),
        ("Ist das Licht in {area} an?", {"domain": "light"}),
        ("Brennt das Licht in {area}", {"domain": "light"}),
        ("Brennt das Licht in {area}?", {"domain": "light"}),
    ],
    "HassSetPosition": [
        ("Öffne die Rollläden in {area}", {"domain": "cover", "command": "open"}),
        ("Schließe die Rollläden in {area}", {"domain": "cover", "command": "close"}),
        ("Rollos {area} hoch", {"domain": "cover", "command": "open"}),
        ("Rollos {area} runter", {"domain": "cover", "command": "close"}),
        ("Fahre alle Rollläden hoch", {"domain": "cover", "command": "open"}),  # Global
        ("Fahre alle Rollläden runter", {"domain": "cover", "command": "close"}),  # Global
    ],
}


def generate_dummy_embedding(dim: int = 1024) -> List[float]:
    """Generate a deterministic dummy embedding based on random but seeded values."""
    # Use a consistent seed for reproducibility
    np.random.seed(hash("semantic_cache_test") % 2**32)
    emb = np.random.randn(dim).astype(np.float32)
    # Normalize
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


def generate_test_anchors() -> Dict[str, Any]:
    """Generate a self-contained test anchor file.
    
    Returns a dict with 'version' and 'anchors' keys, ready to be
    written to JSON or used directly in tests.
    """
    anchors = []
    seen_texts = set()
    
    for intent, patterns in ANCHOR_PATTERNS.items():
        for pattern_template, base_slots in patterns:
            # Check if pattern has {area} placeholder
            if "{area}" in pattern_template:
                # Generate for each test area
                for area in TEST_AREAS:
                    text = pattern_template.format(area=area)
                    if text not in seen_texts:
                        seen_texts.add(text)
                        slots = base_slots.copy()
                        slots["area"] = area
                        anchors.append({
                            "text": text,
                            "embedding": generate_dummy_embedding(),
                            "intent": intent,
                            "entity_ids": [],
                            "slots": slots,
                            "required_disambiguation": False,
                            "disambiguation_options": None,
                            "hits": 0,
                            "last_hit": "",
                            "verified": True,
                            "generated": True,
                        })
            else:
                # Global pattern (no area)
                text = pattern_template
                if text not in seen_texts:
                    seen_texts.add(text)
                    anchors.append({
                        "text": text,
                        "embedding": generate_dummy_embedding(),
                        "intent": intent,
                        "entity_ids": [],
                        "slots": base_slots.copy(),
                        "required_disambiguation": False,
                        "disambiguation_options": None,
                        "hits": 0,
                        "last_hit": "",
                        "verified": True,
                        "generated": True,
                    })
    
    return {
        "version": 2,
        "anchors": anchors,
    }


def extract_test_data_from_anchors(anchors_path: Path) -> Dict[str, Any]:
    """Extract available areas, intents, and entities from an anchor file.
    
    Returns a dict with:
        - areas: List of area names
        - intents: List of intent names
        - entities: List of entity_id strings
        - sample_anchors: Dict mapping intent -> list of sample anchors
    """
    with open(anchors_path) as f:
        data = json.load(f)
    
    areas = set()
    intents = set()
    entities = set()
    sample_anchors = {}
    
    for anchor in data.get("anchors", []):
        intent = anchor.get("intent")
        if intent:
            intents.add(intent)
            if intent not in sample_anchors:
                sample_anchors[intent] = []
            if len(sample_anchors[intent]) < 5:  # Keep up to 5 samples per intent
                sample_anchors[intent].append(anchor)
        
        slots = anchor.get("slots", {})
        if "area" in slots:
            areas.add(slots["area"])
        
        for eid in anchor.get("entity_ids", []):
            entities.add(eid)
    
    return {
        "areas": sorted(areas),
        "intents": sorted(intents),
        "entities": sorted(entities),
        "sample_anchors": sample_anchors,
    }


def get_available_areas() -> List[str]:
    """Get the list of test areas for parametrized tests."""
    return TEST_AREAS.copy()


def get_available_intents() -> List[str]:
    """Get the list of test intents."""
    return list(ANCHOR_PATTERNS.keys())


# Generate test anchors on module load for quick access
SYNTHETIC_ANCHORS = generate_test_anchors()
