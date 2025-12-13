"""Entity keyword constants for natural language understanding.

Contains German keyword mappings for various entity types used in
entity recognition and plural detection.
"""

from typing import Dict, List

# --- DOMAIN-SPECIFIC KEYWORDS ---

LIGHT_KEYWORDS: Dict[str, str] = {
    "licht": "lichter",
    "lampe": "lampen",
    "leuchte": "leuchten",
    "beleuchtung": "beleuchtungen",
    "spot": "spots",
}

COVER_KEYWORDS: Dict[str, str] = {
    "rollladen": "rollläden",
    "rollo": "rollos",
    "jalousie": "jalousien",
    "markise": "markisen",
    "beschattung": "beschattungen",
}

SWITCH_KEYWORDS: Dict[str, str] = {
    "schalter": "schalter",  # Canonical name first
    "steckdose": "steckdosen",
    "zwischenstecker": "zwischenstecker",
    "strom": "strom",
}

FAN_KEYWORDS: Dict[str, str] = {
    "ventilator": "ventilatoren",
    "luefter": "luefter",
    "lüfter": "lüfter",
}

MEDIA_KEYWORDS: Dict[str, str] = {
    "tv": "tvs",
    "fernseher": "fernseher",
    "musik": "musik",
    "radio": "radios",
    "lautsprecher": "lautsprecher",
    "player": "player",
}

SENSOR_KEYWORDS: Dict[str, str] = {
    "sensor": "sensoren",  # Canonical name first
    "temperatur": "temperaturen",
    "luftfeuchtigkeit": "luftfeuchtigkeiten",
    "feuchtigkeit": "feuchtigkeiten",
    "wert": "werte",
    "status": "status",
    "zustand": "zustände",
}

CLIMATE_KEYWORDS: Dict[str, str] = {
    "thermostat": "thermostate",  # Canonical name first
    "heizung": "heizungen",
    "klimaanlage": "klimaanlagen",
}

VACUUM_KEYWORDS: List[str] = [
    "staubsauger",  # Canonical name first
    "saugen",
    "sauge",
    "staubsaugen",
    "staubsauge",
    "wischen",
    "wische",
    "putzen",
    "putze",
    "reinigen",
    "reinige",
    "roboter",
]

TIMER_KEYWORDS: List[str] = [
    "timer",
    "wecker",
    "countdown",
    "uhr",
    "stoppuhr",
]

CALENDAR_KEYWORDS: List[str] = [
    "kalender",
    "termin",
    "termine",
    "ereignis",
    "event",
    "veranstaltung",
    "eintrag",
    "kalendereintrag",
]

OTHER_ENTITY_PLURALS: Dict[str, str] = {
    "das fenster": "die fenster",
    "tür": "türen",
    "tor": "tore",
    "gerät": "geräte",
}

# --- DOMAIN NAME MAPPING ---

# Auto-generate domain names from first keyword in each domain dict
# First keyword is the canonical/response name (e.g., "schalter" not "steckdose")
DOMAIN_NAMES: Dict[str, str] = {
    "light": next(iter(LIGHT_KEYWORDS)).capitalize(),
    "cover": next(iter(COVER_KEYWORDS)).capitalize(),
    "switch": next(iter(SWITCH_KEYWORDS)).capitalize(),
    "fan": next(iter(FAN_KEYWORDS)).capitalize(),
    "climate": next(iter(CLIMATE_KEYWORDS)).capitalize(),
    "media_player": "Mediaplayer",  # Special case - two words
    "vacuum": VACUUM_KEYWORDS[0].capitalize(),  # List, not dict
    "sensor": next(iter(SENSOR_KEYWORDS)).capitalize(),
}


# --- COMBINED MAPPINGS ---

# Combined entity plural mapping (for backward compatibility)
_ENTITY_PLURALS: Dict[str, str] = {
    **LIGHT_KEYWORDS,
    **COVER_KEYWORDS,
    **SWITCH_KEYWORDS,
    **FAN_KEYWORDS,
    **MEDIA_KEYWORDS,
    **SENSOR_KEYWORDS,
    **CLIMATE_KEYWORDS,
    **OTHER_ENTITY_PLURALS,
}

# Export for backward compatibility
ENTITY_PLURALS = _ENTITY_PLURALS
