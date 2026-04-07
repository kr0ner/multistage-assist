"""Self-contained test fixtures for semantic cache tests.

This module generates a minimal, synthetic anchor file for testing
that is INDEPENDENT of any user's Home Assistant installation.
"""

import json
import numpy as np
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Coroutine
from multistage_assist.utils.semantic_cache_builder import CACHE_VERSION
from multistage_assist.utils.german_utils import (
    normalize_for_cache, 
    get_prepositional_area
)
from multistage_assist.utils.cache_patterns.base import (
    DOMAIN_DEVICE_WORDS,
    DOMAIN_DEVICE_WORDS_SINGULAR,
    DOMAIN_DEVICE_WORDS_NOMINATIVE
)
from multistage_assist.constants.area_keywords import AREA_ALIASES


# =============================================================================
# SYNTHETIC TEST DATA - Derived from Central Constants
# =============================================================================

# Reuse area names from central constants to avoid redefinition
TEST_AREAS = sorted(list(set(AREA_ALIASES.values())))
# Floors can be derived if they were in a central place, otherwise we use these standard ones
TEST_FLOORS = ["Erdgeschoss", "Obergeschoss", "Keller", "Dachgeschoss"]

# REFINED PRINCIPLE: Max 2 templates per domain per intent.
# Using {device} and {device_nom} which are resolved via DOMAIN_DEVICE_WORDS constants.
# We now use natural language centroids that match the normalization in german_utils.py.
ANCHOR_PATTERNS = {
    "HassTurnOn": [
        ("Schalte {device} {area_prep} an", {"domain": "light"}),
        ("{device} {area} an", {"domain": "light"}),
        ("Öffne {device} {area_prep}", {"domain": "cover"}),
        ("{device} {area} hoch", {"domain": "cover"}),
        ("Alle {device} an", {"domain": "light"}),
        ("Alle {device} hoch", {"domain": "cover"}),
    ],
    "HassTurnOff": [
        ("Schalte {device} {area_prep} aus", {"domain": "light"}),
        ("{device} {area} aus", {"domain": "light"}),
        ("Schließe {device} {area_prep}", {"domain": "cover"}),
        ("{device} {area} runter", {"domain": "cover"}),
        ("Alle {device} aus", {"domain": "light"}),
        ("Alle {device} runter", {"domain": "cover"}),
    ],
    "HassLightSet": [
        ("Stelle {device} {area_prep} auf 50 Prozent", {"domain": "light", "brightness": 50}),
        ("Mache {device} {area_prep} heller", {"domain": "light", "command": "step_up"}),
        ("{device} {area} heller", {"domain": "light", "command": "step_up"}),
        ("{device} {area} dunkler", {"domain": "light", "command": "step_down"}),
        ("Dimme alle {device}", {"domain": "light"}),
    ],
    "HassGetState": [
        ("Ist {device_nom} {area_prep} an?", {"domain": "light"}),
        ("Wie warm ist es {area_prep}?", {"domain": "climate", "device_class": "temperature"}),
    ],
    "HassSetPosition": [
        ("Fahre {device} {area_prep} auf 50 Prozent", {"domain": "cover", "position": 50}),
        ("Fahre {device} {area_prep} hoch", {"domain": "cover", "command": "open"}),
        ("{device} {area} hoch", {"domain": "cover", "command": "open"}),
        ("{device} {area} runter", {"domain": "cover", "command": "close"}),
        ("Alle {device} auf 50 Prozent", {"domain": "cover", "position": 50}),
    ],
    "HassClimateSetTemperature": [
        ("Stelle die Heizung {area_prep} auf 21 Grad", {"domain": "climate", "temperature": 21}),
        ("Mache die Heizung {area_prep} wärmer", {"domain": "climate", "command": "step_up"}),
    ],
    # 7. ADDED: Unique Entity Commands (No Area)
    "HassTurnOn:unique": [
        ("Schalte {device} {entity_name} an", {"domain": "light"}),
        ("Öffne {device} {entity_name}", {"domain": "cover"}),
    ],
    "HassTurnOff:unique": [
        ("Schalte {device} {entity_name} aus", {"domain": "light"}),
        ("Schließe {device} {entity_name}", {"domain": "cover"}),
    ],
    # 8. ADDED: Complaint-Driven Intents
    "HassLightSet:complaint": [
        ("Es ist zu dunkel {area_prep}", {"domain": "light", "command": "step_up"}),
        ("Es ist zu hell {area_prep}", {"domain": "light", "command": "step_down"}),
    ],
    "HassLightSet:complaint_unique": [
        ("{entity_name} ist zu dunkel", {"domain": "light", "command": "step_up"}),
        ("{entity_name} ist zu hell", {"domain": "light", "command": "step_down"}),
    ],
}


def generate_dummy_embedding(dim: int = 384) -> List[float]:
    """Generate a deterministic dummy embedding based on seed."""
    np.random.seed(hash("semantic_cache_test") % 2**32)
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


async def generate_test_anchors(
    embed_func: Optional[Callable[[str], Coroutine[Any, Any, Optional[np.ndarray]]]] = None,
    areas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate minimal synthetic anchors by REUSING central plural/singular constants."""
    anchors = []
    seen_texts = set()
    
    # Use provided areas or default + some multi-word test cases
    target_areas = list(areas) if areas else list(TEST_AREAS)
    if "Kinder Badezimmer" not in target_areas:
        target_areas.append("Kinder Badezimmer")
    if "Nora s Zimmer" not in target_areas:
        target_areas.append("Nora s Zimmer")
    
    
    for intent, patterns in ANCHOR_PATTERNS.items():
        # Handle regular and :complaint (area-based)
        if ":unique" in intent or ":complaint_unique" in intent:
            continue
            
        for pattern_template, base_slots in patterns:
            domain = base_slots.get("domain", "light")
            
            # REUSE central plural/singular definitions
            d_plural = DOMAIN_DEVICE_WORDS.get(domain)
            d_singular = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain)
            
            if not d_plural or not d_singular:
                continue
                
            variants = [d_plural]
            if d_singular != d_plural:
                variants.append(d_singular)
                
            for device_word in variants:
                device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, device_word)
                
                for area in target_areas:
                    full_prep_phrase = get_prepositional_area(area)
                    # Extract just the prepositional part (e.g., "im" from "im Badezimmer")
                    area_prep = full_prep_phrase.rsplit(area, 1)[0].strip()
                    
                    text = pattern_template.replace("{device}", device_word)
                    text = text.replace("{device_nom}", device_nom)
                    text = text.replace("{area_prep}", area_prep).replace("{area}", area)
                    
                    # Normalize text as we would in production
                    text_norm, _ = normalize_for_cache(text)
                    
                    if text_norm in seen_texts:
                        continue
                    seen_texts.add(text_norm)
                    
                    if embed_func:
                        emb_arr = await embed_func(text_norm)
                        embedding = emb_arr.tolist() if emb_arr is not None else generate_dummy_embedding()
                    else:
                        embedding = generate_dummy_embedding()
                    
                    slots = dict(base_slots)
                    slots["area"] = area
                    
                    anchors.append({
                        "text": text_norm,
                        "intent": intent.split(":")[0], # Strip :unique suffix
                        "slots": slots,
                        "embedding": embedding,
                        "entity_ids": [],
                        "required_disambiguation": False,
                        "disambiguation_options": None,
                        "hits": 0,
                        "last_hit": "",
                        "verified": True,
                        "is_anchor": True,
                        "generated": True
                    })

    # NEW: Handle :unique and :complaint_unique patterns (No area)
    for intent, patterns in ANCHOR_PATTERNS.items():
        if ":unique" not in intent and ":complaint_unique" not in intent:
            continue
            
        for pattern_template, base_slots in patterns:
            domain = base_slots.get("domain", "light")
            d_singular = DOMAIN_DEVICE_WORDS_SINGULAR.get(domain, "das Gerät")
            device_nom = DOMAIN_DEVICE_WORDS_NOMINATIVE.get(domain, d_singular)
            
            # Test entities (including multi-word ones)
            test_entities = [
                ("Deckenlicht", "light"),
                ("Dusche", "light"),
                ("Rollladen", "cover"),
                ("Kinder Badezimmer Licht", "light"), # Multi-word test
                ("Nora s Zimmer Rollladen", "cover"), # Multi-word test
            ]
            
            for entity_name, ent_domain in test_entities:
                if ent_domain != domain:
                    continue
                
                # Apply domain prefixing as production would do
                domain_noun = d_singular.split()[-1] if " " in d_singular else ""
                if domain_noun.lower() in entity_name.lower():
                    prefix = ""
                else:
                    prefix = f" {domain_noun}"
                
                article_acc = d_singular.split()[0] if " " in d_singular else ""
                full_dev_acc = f"{article_acc}{prefix}".strip()
                
                text = pattern_template.replace("{device}", full_dev_acc)
                text = text.replace("{device_nom}", device_nom)
                text = text.replace("{entity_name}", entity_name)
                
                # Track multi-word names for underscore normalization in the test too
                text_norm, _ = normalize_for_cache(text)
                
                if text_norm in seen_texts:
                    continue
                seen_texts.add(text_norm)
                
                if embed_func:
                    emb_arr = await embed_func(text_norm)
                    embedding = emb_arr.tolist() if emb_arr is not None else generate_dummy_embedding()
                else:
                    embedding = generate_dummy_embedding()
                
                slots = dict(base_slots)
                slots["name"] = entity_name
                
                anchors.append({
                    "text": text_norm,
                    "intent": intent.split(":")[0],
                    "slots": slots,
                    "embedding": embedding,
                    "entity_ids": ["test.entity_id"],
                    "required_disambiguation": False,
                    "disambiguation_options": None,
                    "hits": 0,
                    "last_hit": "",
                    "verified": True,
                    "is_anchor": True,
                    "generated": True
                })
                
    return {"version": CACHE_VERSION, "anchors": anchors}


def extract_test_data_from_anchors(anchor_file: Path) -> Dict[str, Any]:
    """Extract list of areas and intents from an existing anchor file."""
    try:
        with open(anchor_file, "r") as f:
            data = json.load(f)
        areas = set()
        intents = set()
        for anchor in data.get("anchors", []):
            intents.add(anchor.get("intent"))
            slots = anchor.get("slots", {})
            if slots.get("area"):
                areas.add(slots["area"])
            elif slots.get("floor"):
                areas.add(slots["floor"])
        return {
            "areas": sorted(list(areas)),
            "intents": sorted(list(intents))
        }
    except Exception:
        return {"areas": TEST_AREAS, "intents": list(ANCHOR_PATTERNS.keys())}
