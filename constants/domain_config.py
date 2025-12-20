"""Centralized domain configuration for MultiStage Assist.

This module provides a single source of truth for domain-related configuration,
enabling easy extension to new domains and future multi-language support.

Domain configuration includes:
- German display names (name_de)
- Device words for semantic patterns (device_word_de)
- Keywords for domain detection
- Supported intents per domain
- Step configuration for relative adjustments

Usage:
    from .domain_config import DOMAIN_CONFIG, get_domain_name
    
    config = DOMAIN_CONFIG["light"]
    intents = config["intents"]
"""

from typing import Any, Dict, List, Optional

# Import existing keyword definitions for backward compatibility
from .entity_keywords import (
    LIGHT_KEYWORDS,
    COVER_KEYWORDS,
    SWITCH_KEYWORDS,
    FAN_KEYWORDS,
    MEDIA_KEYWORDS,
    SENSOR_KEYWORDS,
    CLIMATE_KEYWORDS,
    VACUUM_KEYWORDS,
    TIMER_KEYWORDS,
    CALENDAR_KEYWORDS,
    AUTOMATION_KEYWORDS,
)


def _extract_nouns(keywords_dict: Dict[str, str]) -> List[str]:
    """Extract nouns from 'article noun' format keywords.
    
    E.g., {"das licht": "die lichter"} -> ["licht", "lichter"]
    """
    nouns = []
    for key, value in keywords_dict.items():
        nouns.append(key.split()[-1])   # Last word of key
        nouns.append(value.split()[-1]) # Last word of value
    return nouns


def _get_name_de(keywords_dict: Dict[str, str]) -> str:
    """Get German singular name from first keyword.
    
    E.g., {"das licht": ...} -> "Licht"
    """
    first_key = next(iter(keywords_dict.keys()))
    return first_key.split()[-1].capitalize()


def _get_name_de_plural(keywords_dict: Dict[str, str]) -> str:
    """Get German plural name from first keyword's value.
    
    E.g., {...: "die lichter"} -> "Lichter"
    """
    first_value = next(iter(keywords_dict.values()))
    return first_value.split()[-1].capitalize()


def _get_device_word_de(keywords_dict: Dict[str, str]) -> str:
    """Get German device word with article from first keyword.
    
    Converts to accusative case for use in commands (e.g., "Öffne den Rollladen").
    E.g., {"der rollladen": ...} -> "den Rollladen"
    """
    first_key = next(iter(keywords_dict.keys()))
    parts = first_key.split()
    article = parts[0]
    noun = parts[-1].capitalize()
    # Convert nominative to accusative (der -> den for masculine)
    if article == "der":
        article = "den"
    return article + " " + noun


def _get_device_word_de_plural(keywords_dict: Dict[str, str]) -> str:
    """Get German device word plural with article (accusative case).
    
    E.g., {...: "die lichter"} -> "die Lichter"
    """
    first_value = next(iter(keywords_dict.values()))
    parts = first_value.split()
    # Plural articles stay the same in accusative
    return parts[0] + " " + parts[-1].capitalize()


# --- Domain Configuration ---

DOMAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "light": {
        # Display names derived from LIGHT_KEYWORDS
        "name_de": _get_name_de(LIGHT_KEYWORDS),
        "name_de_plural": _get_name_de_plural(LIGHT_KEYWORDS),
        "device_word_de": _get_device_word_de(LIGHT_KEYWORDS),
        
        # Keywords for domain detection
        "keywords": _extract_nouns(LIGHT_KEYWORDS),
        
        # Supported intents
        "intents": [
            "HassTurnOn",
            "HassTurnOff", 
            "HassLightSet",
            "HassGetState",
            "HassTemporaryControl",
        ],
        
        # Step configuration for relative adjustments
        "step": {
            "attribute": "brightness",
            "step_percent": 35,  # Percentage of current value
            "min_step": 10,      # Minimum absolute step
            "off_to_on": 50,     # Value when turning on from off
            "unit": "%",
        },
        
        # State descriptions for responses
        "states_de": {"on": "an", "off": "aus"},
    },
    
    "cover": {
        # Display names derived from COVER_KEYWORDS
        "name_de": _get_name_de(COVER_KEYWORDS),
        "name_de_plural": _get_name_de_plural(COVER_KEYWORDS),
        "device_word_de": _get_device_word_de_plural(COVER_KEYWORDS),  # Plural for area scope
        "device_word_de_singular": _get_device_word_de(COVER_KEYWORDS),  # Singular for entity scope
        
        "keywords": _extract_nouns(COVER_KEYWORDS),
        
        "intents": [
            "HassTurnOn",  # Open
            "HassTurnOff", # Close
            "HassSetPosition",
            "HassGetState",
            "HassTemporaryControl",
        ],
        
        "step": {
            "attribute": "position",
            "step_percent": 25,
            "min_step": 10,
            "off_to_on": 100,  # Fully open
            "unit": "%",
        },
        
        "states_de": {"open": "offen", "closed": "geschlossen", "opening": "öffnet", "closing": "schließt"},
    },
    
    "switch": {
        "name_de": _get_name_de(SWITCH_KEYWORDS),
        "name_de_plural": _get_name_de_plural(SWITCH_KEYWORDS),
        "device_word_de": _get_device_word_de(SWITCH_KEYWORDS),
        
        "keywords": _extract_nouns(SWITCH_KEYWORDS),
        
        "intents": [
            "HassTurnOn",
            "HassTurnOff",
            "HassGetState",
            "HassTemporaryControl",
        ],
        
        "step": None,  # No step support
        
        "states_de": {"on": "an", "off": "aus"},
    },
    
    "fan": {
        "name_de": _get_name_de(FAN_KEYWORDS),
        "name_de_plural": _get_name_de_plural(FAN_KEYWORDS),
        "device_word_de": _get_device_word_de(FAN_KEYWORDS),
        
        "keywords": _extract_nouns(FAN_KEYWORDS),
        
        "intents": [
            "HassTurnOn",
            "HassTurnOff",
            "HassGetState",
            "HassTemporaryControl",
        ],
        
        "step": {
            "attribute": "percentage",
            "step_percent": 25,
            "min_step": 10,
            "off_to_on": 50,
            "unit": "%",
        },
        
        "states_de": {"on": "an", "off": "aus"},
    },
    
    "climate": {
        "name_de": _get_name_de(CLIMATE_KEYWORDS),
        "name_de_plural": _get_name_de_plural(CLIMATE_KEYWORDS),
        "device_word_de": _get_device_word_de(CLIMATE_KEYWORDS),
        
        "keywords": _extract_nouns(CLIMATE_KEYWORDS),
        
        "intents": [
            "HassClimateSetTemperature",
            "HassTurnOn",
            "HassTurnOff",
            "HassGetState",
        ],
        
        "step": {
            "attribute": "temperature",
            "step_absolute": 1.0,  # Absolute step (not percentage)
            "min_temp": 16,
            "max_temp": 28,
            "unit": "°C",
        },
        
        "states_de": {"heat": "heizt", "cool": "kühlt", "off": "aus", "idle": "im Leerlauf"},
    },
    
    "media_player": {
        "name_de": _get_name_de(MEDIA_KEYWORDS),
        "name_de_plural": _get_name_de_plural(MEDIA_KEYWORDS),
        "device_word_de": _get_device_word_de(MEDIA_KEYWORDS),
        
        "keywords": _extract_nouns(MEDIA_KEYWORDS),
        
        "intents": [
            "HassTurnOn",
            "HassTurnOff",
            "HassGetState",
        ],
        
        "step": None,
        
        "states_de": {"on": "an", "off": "aus", "playing": "spielt", "paused": "pausiert", "idle": "im Leerlauf"},
    },
    
    "sensor": {
        "name_de": _get_name_de(SENSOR_KEYWORDS),
        "name_de_plural": _get_name_de_plural(SENSOR_KEYWORDS),
        "device_word_de": _get_device_word_de(SENSOR_KEYWORDS),
        
        "keywords": _extract_nouns(SENSOR_KEYWORDS) + ["grad", "warm", "kalt", "wieviel"],
        
        "intents": ["HassGetState"],
        
        "step": None,
        
        "states_de": {},  # Sensors have variable states
    },
    
    "vacuum": {
        "name_de": "Staubsauger",
        "name_de_plural": "Staubsauger",
        "device_word_de": "den Staubsauger",
        
        "keywords": VACUUM_KEYWORDS,
        
        "intents": ["HassVacuumStart"],
        
        "step": None,
        
        "states_de": {"cleaning": "saugt", "docked": "angedockt", "returning": "kehrt zurück", "idle": "im Leerlauf"},
    },
    
    "timer": {
        "name_de": "Timer",
        "name_de_plural": "Timer",
        "device_word_de": "den Timer",
        
        "keywords": TIMER_KEYWORDS,
        
        "intents": ["HassTimerSet"],
        
        "step": None,
        
        "states_de": {},
    },
    
    "calendar": {
        "name_de": "Kalender",
        "name_de_plural": "Kalender",
        "device_word_de": "den Kalender",
        
        "keywords": CALENDAR_KEYWORDS,
        
        "intents": ["HassCalendarCreate", "HassCreateEvent"],
        
        "step": None,
        
        "states_de": {},
    },
    
    "automation": {
        "name_de": "Automatisierung",
        "name_de_plural": "Automatisierungen",
        "device_word_de": "die Automatisierung",
        
        "keywords": AUTOMATION_KEYWORDS,
        
        "intents": [
            "HassTurnOn",
            "HassTurnOff",
            "HassTemporaryControl",
        ],
        
        "step": None,
        
        "states_de": {"on": "aktiv", "off": "inaktiv"},
    },
}


# --- Floor Aliases (German abbreviations and synonyms) ---
FLOOR_ALIASES_DE: Dict[str, List[str]] = {
    "eg": ["erdgeschoss", "ground floor", "parterre"],
    "erdgeschoss": ["eg", "ground floor", "parterre", "unten"],
    "og": ["obergeschoss", "first floor", "oben"],
    "obergeschoss": ["og", "first floor", "oben", "1og", "1. og"],
    "ug": ["untergeschoss", "basement", "keller"],
    "untergeschoss": ["ug", "basement", "keller"],
    "keller": ["ug", "untergeschoss", "basement"],
    "dg": ["dachgeschoss", "attic"],
    "dachgeschoss": ["dg", "attic", "dach"],
}


# --- Helper Functions ---

def get_domain_name(domain: str, plural: bool = False) -> str:
    """Get German name for a domain.
    
    Args:
        domain: Entity domain (e.g., "light")
        plural: Whether to return plural form
        
    Returns:
        German domain name
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return domain.title()
    
    if plural:
        return config.get("name_de_plural", config.get("name_de", domain.title()))
    return config.get("name_de", domain.title())


def get_device_word(domain: str) -> str:
    """Get German device word for semantic patterns.
    
    Args:
        domain: Entity domain
        
    Returns:
        Device word with article (e.g., "das Licht")
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return f"das {domain}"
    return config.get("device_word_de", f"das {domain}")


def get_domain_keywords(domain: str) -> List[str]:
    """Get keywords for domain detection.
    
    Args:
        domain: Entity domain
        
    Returns:
        List of German keywords
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return []
    return config.get("keywords", [])


def get_domain_intents(domain: str) -> List[str]:
    """Get supported intents for a domain.
    
    Args:
        domain: Entity domain
        
    Returns:
        List of intent names
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return []
    return config.get("intents", [])


def get_step_config(domain: str) -> Optional[Dict[str, Any]]:
    """Get step configuration for relative adjustments.
    
    Args:
        domain: Entity domain
        
    Returns:
        Step config dict or None if not supported
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return None
    return config.get("step")


def get_state_description(domain: str, state: str) -> str:
    """Get German description for an entity state.
    
    Args:
        domain: Entity domain
        state: State value (e.g., "on", "off")
        
    Returns:
        German state description
    """
    config = DOMAIN_CONFIG.get(domain)
    if not config:
        return state
    states = config.get("states_de", {})
    return states.get(state, state)


def detect_domain_from_text(text: str) -> Optional[str]:
    """Detect domain from German text using keywords.
    
    Args:
        text: Input text
        
    Returns:
        Detected domain or None
    """
    text_lower = text.lower()
    matches = []
    
    for domain, config in DOMAIN_CONFIG.items():
        keywords = config.get("keywords", [])
        if any(kw in text_lower for kw in keywords):
            matches.append(domain)
    
    if len(matches) == 1:
        return matches[0]
    
    # Handle common conflicts
    if "climate" in matches and "sensor" in matches:
        return "climate"
    if "calendar" in matches:
        return "calendar"
    if "timer" in matches:
        return "timer"
    if "vacuum" in matches:
        return "vacuum"
    
    return matches[0] if matches else None
