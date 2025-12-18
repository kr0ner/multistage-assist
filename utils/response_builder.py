"""Response building utilities for German language output.

Provides helper functions for building natural German responses including
entity lists, confirmation messages, questions, and error messages.
"""

import re
from typing import Any, Dict, List, Optional


# --- Entity List Formatting ---

def join_names(names: List[str], conjunction: str = "und") -> str:
    """Format list of names with German conjunction.
    
    Args:
        names: List of entity/device names
        conjunction: Conjunction word (default: "und")
        
    Returns:
        Formatted string: "A, B und C"
        
    Examples:
        join_names(["Küche", "Bad"]) -> "Küche und Bad"
        join_names(["A", "B", "C"]) -> "A, B und C"
        join_names(["Küche"]) -> "Küche"
        join_names([]) -> ""
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} {conjunction} {names[-1]}"


def format_entity_list(entities: List[str], max_display: int = 5) -> str:
    """Format list of entities for display with truncation.
    
    Args:
        entities: List of entity names
        max_display: Maximum entities to show before truncating
        
    Returns:
        Formatted string with optional "und X weitere"
        
    Examples:
        format_entity_list(["A", "B"]) -> "A und B"
        format_entity_list(["A", "B", "C", "D", "E", "F"]) -> "A, B, C, D, E und 1 weiteres"
    """
    if not entities:
        return ""
    
    if len(entities) <= max_display:
        return join_names(entities)
    
    shown = entities[:max_display]
    remaining = len(entities) - max_display
    suffix = "weiteres" if remaining == 1 else "weitere"
    
    return f"{join_names(shown)} und {remaining} {suffix}"


# --- Confirmation Messages ---

DOMAIN_NAMES_DE = {
    "light": "Licht",
    "cover": "Rollo",
    "switch": "Schalter",
    "fan": "Ventilator",
    "climate": "Heizung",
    "media_player": "Media Player",
    "sensor": "Sensor",
    "vacuum": "Staubsauger",
    "automation": "Automatisierung",
}

INTENT_VERBS_DE = {
    "HassTurnOn": "an",
    "HassTurnOff": "aus",
    "HassLightSet": "eingestellt",
    "HassSetPosition": "eingestellt",
    "HassClimateSetTemperature": "eingestellt",
    "HassGetState": "abgefragt",
    "HassVacuumStart": "gestartet",
}


def get_domain_name(domain: str) -> str:
    """Get German name for entity domain.
    
    Args:
        domain: Entity domain (e.g., "light", "cover")
        
    Returns:
        German domain name
    """
    return DOMAIN_NAMES_DE.get(domain, domain.title())


def build_confirmation(
    intent_name: str,
    device_names: List[str],
    domain: str = None,
    params: Dict[str, Any] = None,
) -> str:
    """Build a simple German confirmation message.
    
    Args:
        intent_name: HA intent name
        device_names: List of affected device names
        domain: Entity domain (optional)
        params: Additional parameters (brightness, position, etc.)
        
    Returns:
        German confirmation message
        
    Examples:
        build_confirmation("HassTurnOn", ["Küche"]) -> "Küche ist an."
        build_confirmation("HassLightSet", ["Büro"], params={"brightness": 50}) 
            -> "Büro ist auf 50% gesetzt."
    """
    if not device_names:
        return "Aktion ausgeführt."
    
    devices = join_names(device_names)
    params = params or {}
    
    # Handle specific intents
    if intent_name == "HassTurnOn":
        return f"{devices} ist an."
    
    if intent_name == "HassTurnOff":
        return f"{devices} ist aus."
    
    if intent_name == "HassLightSet":
        if "brightness" in params:
            return f"{devices} ist auf {params['brightness']}% gesetzt."
        return f"{devices} ist eingestellt."
    
    if intent_name == "HassSetPosition":
        if "position" in params:
            return f"{devices} ist auf {params['position']}% gesetzt."
        return f"{devices} ist eingestellt."
    
    if intent_name == "HassClimateSetTemperature":
        if "temperature" in params:
            return f"{devices} ist auf {params['temperature']}° gesetzt."
        return f"{devices} ist eingestellt."
    
    if intent_name == "HassTemporaryControl":
        duration = params.get("duration_str", "")
        action = params.get("action", "eingestellt")
        if duration:
            return f"{devices} ist für {duration} {action}."
        return f"{devices} ist temporär {action}."
    
    if intent_name == "HassVacuumStart":
        mode = params.get("mode", "")
        area = params.get("area", "")
        if mode == "mop":
            verb = "wischt"
        else:
            verb = "saugt"
        if area:
            return f"Staubsauger {verb} {area}."
        return f"Staubsauger gestartet."
    
    # Generic fallback
    verb = INTENT_VERBS_DE.get(intent_name, "ausgeführt")
    return f"{devices} ist {verb}."


# --- State Query Responses ---

# State descriptions per domain
STATE_DESCRIPTIONS_DE = {
    "light": {"on": "an", "off": "aus"},
    "switch": {"on": "an", "off": "aus"},
    "fan": {"on": "an", "off": "aus"},
    "cover": {"open": "offen", "closed": "geschlossen", "opening": "öffnet", "closing": "schließt"},
    "climate": {"heat": "heizt", "cool": "kühlt", "off": "aus", "idle": "im Leerlauf"},
    "media_player": {"on": "an", "off": "aus", "playing": "spielt", "paused": "pausiert", "idle": "im Leerlauf"},
    "vacuum": {"cleaning": "saugt", "docked": "angedockt", "returning": "kehrt zurück", "idle": "im Leerlauf"},
    "sensor": {},  # Sensors use their value directly
    "binary_sensor": {"on": "aktiv", "off": "inaktiv"},
    "automation": {"on": "aktiv", "off": "inaktiv"},
}


def build_state_response(
    device_names: List[str],
    states: List[str],
    domain: str = None,
) -> str:
    """Build a German state query response.
    
    Args:
        device_names: List of device names that match the query
        states: List of states for each device
        domain: Entity domain (optional, for state translation)
        
    Returns:
        German state response
        
    Examples:
        build_state_response(["Wohnzimmer"], ["on"], "light")
            -> "Wohnzimmer ist an."
        build_state_response(["Wohnzimmer", "Büro"], ["on", "on"], "light")
            -> "Wohnzimmer und Büro sind an."
        build_state_response(["Küche", "Bad", "Büro"], ["on", "off", "on"], "light")
            -> "Küche und Büro sind an. Bad ist aus."
    """
    if not device_names:
        return "Keine Geräte gefunden."
    
    # Get state translations for this domain
    state_map = STATE_DESCRIPTIONS_DE.get(domain, {})
    
    # Group devices by state
    state_groups: Dict[str, List[str]] = {}
    for name, state in zip(device_names, states):
        translated = state_map.get(state, state)
        if translated not in state_groups:
            state_groups[translated] = []
        state_groups[translated].append(name)
    
    # Build response parts
    parts = []
    for state, names in state_groups.items():
        devices = join_names(names)
        verb = "ist" if len(names) == 1 else "sind"
        parts.append(f"{devices} {verb} {state}")
    
    return ". ".join(parts) + "."


# --- Question Building ---

def build_question(field: str, context: str = None) -> str:
    """Build a German question for missing field.
    
    Args:
        field: Field name being requested
        context: Optional context to include
        
    Returns:
        German question string
    """
    questions = {
        "name": "Wie soll es heißen?",
        "summary": "Wie soll der Termin heißen?",
        "title": "Wie soll der Titel lauten?",
        "duration": "Wie lange?",
        "time": "Um wie viel Uhr?",
        "date": "An welchem Tag?",
        "datetime": "Wann soll es sein?",
        "device": "Auf welchem Gerät?",
        "area": "In welchem Bereich?",
        "calendar": "In welchen Kalender?",
    }
    
    return questions.get(field, f"Bitte gib {field} an.")


def build_selection_question(
    field: str,
    options: List[str],
    max_options: int = 5,
) -> str:
    """Build a German selection question with options.
    
    Args:
        field: Field being selected
        options: Available options
        max_options: Max options to show
        
    Returns:
        Question with options list
        
    Example:
        build_selection_question("device", ["Handy", "Tablet"]) 
            -> "Auf welchem Gerät? (Handy, Tablet)"
    """
    base_question = build_question(field)
    
    if not options:
        return base_question
    
    if len(options) <= max_options:
        options_str = ", ".join(options)
    else:
        shown = options[:max_options]
        remaining = len(options) - max_options
        options_str = f"{', '.join(shown)} (+{remaining})"
    
    return f"{base_question} ({options_str})"


# --- Error Messages ---

ERROR_MESSAGES = {
    "not_found": "Das habe ich nicht gefunden.",
    "ambiguous": "Das ist nicht eindeutig.",
    "no_permission": "Dafür habe ich keine Berechtigung.",
    "connection": "Verbindungsproblem.",
    "timeout": "Zeitüberschreitung.",
    "unknown": "Ein Fehler ist aufgetreten.",
    "no_devices": "Keine Geräte gefunden.",
    "no_calendars": "Keine Kalender gefunden.",
    "invalid_date": "Ungültiges Datum.",
    "invalid_time": "Ungültige Zeitangabe.",
    "invalid_duration": "Ungültige Dauer.",
}


def build_error(error_type: str, details: str = None) -> str:
    """Build a German error message.
    
    Args:
        error_type: Error type key
        details: Optional details to append
        
    Returns:
        German error message
    """
    base = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["unknown"])
    
    if details:
        return f"{base} {details}"
    return base


# --- TTS Normalization ---

TTS_REPLACEMENTS = {
    "°C": " Grad Celsius",
    "°": " Grad",
    "%": " Prozent",
    "kWh": " Kilowattstunden",
    "kW": " Kilowatt",
    "W": " Watt",
    "V": " Volt",
    "A": " Ampere",
    "lx": " Lux",
    "lm": " Lumen",
}


def normalize_for_tts(text: str) -> str:
    """Normalize text for text-to-speech.
    
    Converts symbols to spoken words and adjusts number formats.
    
    Args:
        text: Input text
        
    Returns:
        TTS-friendly text
        
    Examples:
        normalize_for_tts("22°C") -> "22 Grad Celsius"
        normalize_for_tts("50%") -> "50 Prozent"
    """
    if not text:
        return ""
    
    # Convert decimal points to commas (German style)
    text = re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)
    
    # Replace symbols
    for sym, spoken in TTS_REPLACEMENTS.items():
        text = text.replace(sym, spoken)
    
    return text
