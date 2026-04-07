"""German language constants for areas and floors.

Provides mappings for room aliases, prepositional hints, and floor indicators.
"""

from typing import Dict, List, Set


# Mapping for room aliases (user name -> canonical name)
AREA_ALIASES: Dict[str, str] = {
    "bad": "Badezimmer",
    "wc": "Gäste-WC",
    "kizi": "Kinderzimmer",
    "sz": "Schlafzimmer",
    "wz": "Wohnzimmer",
    "ez": "Esszimmer",
    "kue": "Küche",
    "fl": "Flur",
    "flur": "Flur",
    "ke": "Keller",
    "buero": "Büro",
    "arbeitszimmer": "Büro",
    "garage": "Garage",
    "carport": "Carport",
    "balkon": "Balkon",
    "terrasse": "Terrasse",
    "garten": "Garten",
    "eingang": "Eingang",
    "haus": "Haus",
    "eg": "Erdgeschoss",
    "og": "Obergeschoss",
    "ug": "Untergeschoss",
}

# Mapping for prepositional hints
AREA_PREPOSITIONS: Dict[str, str] = {
    "küche": "in der",
    "terrasse": "auf der",
    "balkon": "auf dem",
    "garten": "im",
    "eingang": "am",
    "keller": "im",
    "flur": "im",
    "dachboden": "auf dem",
    "garage": "in der",
    "carport": "im",
    "büro": "im",
}

# Area indicators for multi-area detection
AREA_INDICATORS: List[str] = [
    "in der", "im", "in", "auf dem", "auf der",
]

FLOOR_KEYWORDS: List[str] = [
    "geschoss", "stock", "etage", "eg", "og", "ug", "dg",
]

# Location indicators for detecting area references in compound commands
LOCATION_INDICATORS: List[str] = [" in ", " im "]
