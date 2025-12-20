"""Entity keyword constants for natural language understanding.

Contains German keyword mappings for various entity types used in
entity recognition and plural detection.
"""

from typing import Dict, List

# --- DOMAIN-SPECIFIC KEYWORDS (with articles) ---
# Format: "article singular" -> "article plural"
# To get just the noun, split and take index [1] or [-1]

LIGHT_KEYWORDS: Dict[str, str] = {
    "das licht": "die lichter",
    "die lampe": "die lampen",
    "die leuchte": "die leuchten",
    "die beleuchtung": "die beleuchtungen",
    "der spot": "die spots",
}

COVER_KEYWORDS: Dict[str, str] = {
    "der rollladen": "die rollläden",
    "das rollo": "die rollos",
    "die jalousie": "die jalousien",
    "die markise": "die markisen",
    "die beschattung": "die beschattungen",
}

SWITCH_KEYWORDS: Dict[str, str] = {
    "der schalter": "die schalter",
    "die steckdose": "die steckdosen",
    "der zwischenstecker": "die zwischenstecker",
    "der strom": "der strom",
}

FAN_KEYWORDS: Dict[str, str] = {
    "der ventilator": "die ventilatoren",
    "der lüfter": "die lüfter",
}

MEDIA_KEYWORDS: Dict[str, str] = {
    "der tv": "die tvs",
    "der fernseher": "die fernseher",
    "die musik": "die musik",
    "das radio": "die radios",
    "der lautsprecher": "die lautsprecher",
    "der player": "die player",
}

SENSOR_KEYWORDS: Dict[str, str] = {
    "der sensor": "die sensoren",
    "die temperatur": "die temperaturen",
    "die luftfeuchtigkeit": "die luftfeuchtigkeiten",
    "die feuchtigkeit": "die feuchtigkeiten",
    "der wert": "die werte",
    "der status": "die status",
    "der zustand": "die zustände",
}

CLIMATE_KEYWORDS: Dict[str, str] = {
    "das thermostat": "die thermostate",
    "die heizung": "die heizungen",
    "die klimaanlage": "die klimaanlagen",
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
    "stoppuhr",
    # Note: "uhr" removed - too generic, conflicts with time expressions like "15:00 Uhr"
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

AUTOMATION_KEYWORDS: List[str] = [
    "klingel",
    "türklingel",
    "doorbell",
    "benachrichtigung",
    "alarm",
    "automation",
    "automatisierung",
]

OTHER_ENTITY_PLURALS: Dict[str, str] = {
    "das fenster": "die fenster",
    "die tür": "die türen",
    "das tor": "die tore",
    "das gerät": "die geräte",
}

# --- DOMAIN NAME MAPPING ---

# Helper to get noun from "article noun" format
def _get_noun(keyword: str) -> str:
    """Extract noun from 'article noun' format, e.g., 'das licht' -> 'Licht'"""
    return keyword.split()[-1].capitalize()

# Auto-generate domain names from first keyword in each domain dict
# First keyword is the canonical/response name (e.g., "der schalter" -> "Schalter")
DOMAIN_NAMES: Dict[str, str] = {
    "light": _get_noun(next(iter(LIGHT_KEYWORDS))),
    "cover": _get_noun(next(iter(COVER_KEYWORDS))),
    "switch": _get_noun(next(iter(SWITCH_KEYWORDS))),
    "fan": _get_noun(next(iter(FAN_KEYWORDS))),
    "climate": _get_noun(next(iter(CLIMATE_KEYWORDS))),
    "media_player": "Mediaplayer",  # Special case - two words
    "vacuum": VACUUM_KEYWORDS[0].capitalize(),  # List, not dict
    "sensor": _get_noun(next(iter(SENSOR_KEYWORDS))),
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
