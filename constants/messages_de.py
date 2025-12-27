"""Centralized German language messages for MultiStage Assist.

This module provides a single source of truth for all German language strings
used in user-facing responses. This centralizes translations and enables
easier future internationalization.

Message categories:
- ERROR_MESSAGES: Error and failure messages
- QUESTION_TEMPLATES: Questions for missing information
- CONFIRMATION_TEMPLATES: Action confirmation messages  
- SYSTEM_MESSAGES: System-level messages

Usage:
    from .messages_de import ERROR_MESSAGES, get_error_message
    
    msg = get_error_message("not_found")
    # Returns: "Das habe ich nicht gefunden."
"""

from typing import Dict, Optional, Set


# --- Global Scope Keywords ---
# Words that indicate "all areas" or "whole house" scope
GLOBAL_KEYWORDS: Set[str] = {
    "haus",
    "wohnung", 
    "daheim",
    "zuhause",
    "überall",
    "alles",
    "ganze haus",
    "ganzes haus",
    "alle bereiche",
    "alle räume",
}


# --- Selection Keywords ---
# Keywords for all/none selection in disambiguation
ALL_KEYWORDS: Set[str] = {"alle", "alles", "beide", "beiden", "beides"}
NONE_KEYWORDS: Set[str] = {"keine", "keines", "keinen", "nichts", "nein", "nee", "keins"}


# --- German Ordinal Mappings ---
# Maps ordinal words to numeric values. -1 means "last".
ORDINAL_MAP: Dict[str, int] = {
    # Words (all genders/cases)
    "erste": 1, "ersten": 1, "erstes": 1, "erster": 1,
    "zweite": 2, "zweiten": 2, "zweites": 2, "zweiter": 2,
    "dritte": 3, "dritten": 3, "drittes": 3, "dritter": 3,
    "vierte": 4, "vierten": 4, "viertes": 4, "vierter": 4,
    "fünfte": 5, "fünften": 5, "fünftes": 5, "fünfter": 5,
    "sechste": 6, "sechsten": 6, "sechstes": 6, "sechster": 6,
    "siebte": 7, "siebten": 7, "siebtes": 7, "siebter": 7,
    "achte": 8, "achten": 8, "achtes": 8, "achter": 8,
    "neunte": 9, "neunten": 9, "neuntes": 9, "neunter": 9,
    "zehnte": 10, "zehnten": 10, "zehntes": 10, "zehnter": 10,
    "letzte": -1, "letzten": -1, "letztes": -1, "letzter": -1,  # -1 = last
}


# --- Error Messages ---

ERROR_MESSAGES: Dict[str, str] = {
    # General errors
    "not_understood": "Entschuldigung, ich habe das nicht verstanden.",
    "not_found": "Das habe ich nicht gefunden.",
    "unknown_error": "Ein Fehler ist aufgetreten.",
    "action_failed": "Die Aktion konnte nicht ausgeführt werden.",
    
    # Entity/Device errors
    "no_devices": "Keine Geräte gefunden.",
    "entity_not_found": "Dieses Gerät konnte ich nicht finden.",
    "entity_unavailable": "Das Gerät ist nicht verfügbar.",
    "ambiguous_entity": "Das ist nicht eindeutig.",
    
    # Area/Location errors
    "area_not_found": "Diesen Bereich konnte ich nicht finden.",
    "no_devices_in_area": "In diesem Bereich gibt es keine passenden Geräte.",
    
    # Permission/Access errors
    "no_permission": "Dafür habe ich keine Berechtigung.",
    "not_exposed": "Dieses Gerät ist nicht für Sprachbefehle freigegeben.",
    
    # Connection errors
    "connection_error": "Verbindungsproblem.",
    "timeout": "Zeitüberschreitung.",
    "service_unavailable": "Der Dienst ist nicht verfügbar.",
    
    # Validation errors
    "invalid_value": "Ungültiger Wert.",
    "invalid_date": "Ungültiges Datum.",
    "invalid_time": "Ungültige Zeitangabe.",
    "invalid_duration": "Ungültige Dauer.",
    "value_out_of_range": "Der Wert liegt außerhalb des gültigen Bereichs.",
    
    # Calendar errors
    "no_calendars": "Keine Kalender gefunden.",
    "event_creation_failed": "Der Termin konnte nicht erstellt werden.",
    
    # Timer errors
    "timer_failed": "Der Timer konnte nicht gesetzt werden.",
    "no_timer_device": "Kein Gerät für Timer gefunden.",
    
    # Timebox errors
    "timebox_failed": "Fehler beim Ausführen der zeitlichen Steuerung.",
    
    # API errors
    "api_quota_exceeded": "Entschuldigung, der Cloud-Dienst ist vorübergehend nicht erreichbar. Bitte versuche es später erneut.",
    "api_error": "Entschuldigung, bei der Cloud-Anfrage ist ein Fehler aufgetreten.",
    "gemini_unavailable": "Der Cloud-Dienst ist nicht konfiguriert.",
    "no_response": "Entschuldigung, ich habe keine Antwort erhalten.",
}


# --- Question Templates ---

QUESTION_TEMPLATES: Dict[str, str] = {
    # Missing slot questions
    "name": "Wie soll es heißen?",
    "summary": "Wie soll der Termin heißen?",
    "title": "Wie soll der Titel lauten?",
    "duration": "Wie lange?",
    "time": "Um wie viel Uhr?",
    "date": "An welchem Tag?",
    "datetime": "Wann soll es sein?",
    "device": "Auf welchem Gerät?",
    "area": "In welchem Bereich?",
    "room": "In welchem Raum?",
    "calendar": "In welchen Kalender?",
    "entity": "Welches Gerät meinst du?",
    
    # Clarification questions
    "which_one": "Welches meinst du?",
    "which_room": "In welchem Raum?",
    "all_or_specific": "Alle oder nur bestimmte?",
    
    # Confirmation questions
    "confirm_action": "Soll ich das machen?",
    "confirm_learn": "Soll ich mir das merken?",
    "confirm_delete": "Soll ich das löschen?",
    "are_you_sure": "Bist du sicher?",
}


# --- Confirmation Templates ---

CONFIRMATION_TEMPLATES: Dict[str, str] = {
    # Simple confirmations
    "ok": "Okay.",
    "done": "Erledigt.",
    "confirmed": "Alles klar.",
    "noted": "Alles klar, gemerkt.",
    "not_noted": "Okay, nicht gemerkt.",
    "cancelled": "Abgebrochen.",
    
    # Action confirmations (with placeholders)
    "turned_on": "{device} ist an.",
    "turned_off": "{device} ist aus.",
    "set_to": "{device} ist auf {value} gesetzt.",
    "set_brightness": "{device} ist auf {value}% gesetzt.",
    "set_temperature": "{device} ist auf {value}° gesetzt.",
    "set_position": "{device} ist auf {value}% gesetzt.",
    
    # Temporary control
    "temporary_on": "{device} ist für {duration} an.",
    "temporary_off": "{device} ist für {duration} aus.",
    "temporary_set": "{device} ist für {duration} auf {value} gesetzt.",
    
    # Timer confirmations
    "timer_set": "Timer für {duration} gestellt.",
    "timer_named": "Timer '{name}' für {duration} gestellt.",
    
    # Vacuum confirmations
    "vacuum_started": "Staubsauger gestartet.",
    "vacuum_area": "Staubsauger {mode} {area}.",
    
    # Calendar confirmations
    "event_created": "Termin wurde erstellt.",
    "event_preview": "Termin: {summary} am {date} um {time}.",
}


# --- State Query Responses ---

STATE_RESPONSES: Dict[str, str] = {
    # Yes/No responses
    "yes_all_on": "Ja, alle sind an.",
    "yes_all_off": "Ja, alle sind aus.",
    "no_some_on": "Nein, {exceptions} {verb} noch an.",
    "no_some_off": "Nein, {exceptions} {verb} noch aus.",
    "no_some_other": "Nein, {count} sind noch {state}.",
    
    # State reports
    "none_match": "Keine Geräte sind {state}.",
    "state_is": "{device} ist {state}.",
    "states_are": "{devices} sind {state}.",
}


# --- System Messages ---

SYSTEM_MESSAGES: Dict[str, str] = {
    # Initialization messages
    "cache_loading": "Lade Cache...",
    "cache_loaded": "Cache geladen.",
    "cache_generating": "Generiere Cache-Einträge...",
    
    # Learning messages
    "learning_offer": "Soll ich mir merken, dass '{alias}' '{target}' bedeutet?",
    "learning_offer_entity": "Übrigens, ich habe '{src}' als Gerät '{tgt}' interpretiert. Soll ich mir das merken?",
    "learning_offer_area": "Übrigens, ich habe '{src}' als Bereich '{tgt}' interpretiert. Soll ich mir das merken?",
    
    # Disambiguation messages
    "multiple_matches": "Ich habe mehrere passende Geräte gefunden:",
    "please_specify": "Bitte sag mir genauer, welches du meinst.",
    "which_device": "Welches Gerät meinst du?",
    "did_not_understand": "Ich habe leider nichts verstanden.",
    

    # Generic error
    "error_short": "Fehler.",
}


# --- Helper Functions ---

def get_error_message(error_type: str, details: Optional[str] = None) -> str:
    """Get a German error message.
    
    Args:
        error_type: Error type key
        details: Optional details to append
        
    Returns:
        German error message
    """
    base = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["unknown_error"])
    
    if details:
        return f"{base} {details}"
    return base


def get_question(field: str) -> str:
    """Get a German question for a missing field.
    
    Args:
        field: Field name being requested
        
    Returns:
        German question string
    """
    return QUESTION_TEMPLATES.get(field, f"Bitte gib {field} an.")


def get_confirmation(key: str, **kwargs) -> str:
    """Get a German confirmation message with placeholders filled.
    
    Args:
        key: Confirmation template key
        **kwargs: Values to substitute for placeholders
        
    Returns:
        Formatted German confirmation
        
    Example:
        get_confirmation("turned_on", device="Küche")
        # Returns: "Küche ist an."
    """
    template = CONFIRMATION_TEMPLATES.get(key, CONFIRMATION_TEMPLATES["done"])
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def get_state_response(key: str, **kwargs) -> str:
    """Get a German state query response with placeholders filled.
    
    Args:
        key: Response template key
        **kwargs: Values to substitute for placeholders
        
    Returns:
        Formatted German state response
    """
    template = STATE_RESPONSES.get(key, "{device} ist {state}.")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


# --- Domain-Based Response Templates ---
# Each domain/intent has 5 variations for natural variety

import random
from typing import List

# Action verbs for on/off (used in templates)
ACTION_VERBS = {
    "on": "an",
    "off": "aus",
}

# Domain-specific response templates
# {name} = device/area name, {action} = an/aus, {value} = percentage/value
DOMAIN_RESPONSES: Dict[str, Dict[str, List[str]]] = {
    "light": {
        # HassTurnOn / HassTurnOff (combined with {action})
        "toggle": [
            "{name} ist jetzt {action}.",
            "{name} ist {action}.",
            "Ich habe {name} {action}gemacht.",
            "Das Licht in {name} ist {action}.",
            "{name} ist jetzt {action}geschaltet.",
        ],
        "brightness_up": [
            "{name} ist jetzt heller.",
            "Ich habe {name} aufgehellt.",
            "{name} ist heller gestellt.",
            "Die Helligkeit von {name} ist erhöht.",
            "{name} leuchtet jetzt stärker.",
        ],
        "brightness_down": [
            "{name} ist jetzt dunkler.",
            "Ich habe {name} gedimmt.",
            "{name} ist gedimmt.",
            "Die Helligkeit von {name} ist reduziert.",
            "{name} leuchtet jetzt schwächer.",
        ],
        "brightness_set": [
            "{name} ist auf {value}% gestellt.",
            "{name} leuchtet jetzt mit {value}%.",
            "Helligkeit von {name} ist {value}%.",
            "{name} ist auf {value}% eingestellt.",
            "Ich habe {name} auf {value}% gesetzt.",
        ],
        "state_on": [
            "{name} ist an.",
            "{name} ist eingeschaltet.",
            "{name} leuchtet.",
        ],
        "state_off": [
            "{name} ist aus.",
            "{name} ist ausgeschaltet.",
            "{name} leuchtet nicht.",
        ],
    },
    "cover": {
        "toggle": [
            "{name} ist jetzt {action}.",
            "Die Rollläden {name} sind {action}.",
            "Ich habe {name} {action}gemacht.",
            "{name} ist {action}.",
            "Rollläden in {name} sind {action}.",
        ],
        "open": [
            "{name} ist jetzt offen.",
            "Die Rollläden {name} sind offen.",
            "Ich habe {name} geöffnet.",
            "{name} ist hochgefahren.",
            "{name} ist oben.",
        ],
        "close": [
            "{name} ist jetzt geschlossen.",
            "Die Rollläden {name} sind zu.",
            "Ich habe {name} geschlossen.",
            "{name} ist runtergefahren.",
            "{name} ist unten.",
        ],
        "position": [
            "{name} ist auf {value}%.",
            "Ich habe {name} auf {value}% gestellt.",
            "{name} steht jetzt bei {value}%.",
            "Die Position von {name} ist {value}%.",
            "{name} ist auf {value}% eingestellt.",
        ],
        "state_open": [
            "{name} ist offen.",
            "{name} ist geöffnet.",
            "{name} ist oben.",
        ],
        "state_closed": [
            "{name} ist geschlossen.",
            "{name} ist zu.",
            "{name} ist unten.",
        ],
    },
    "switch": {
        "toggle": [
            "{name} ist jetzt {action}.",
            "Die Steckdose {name} ist {action}.",
            "Ich habe {name} {action}geschaltet.",
            "{name} ist {action}.",
            "Strom für {name} ist {action}.",
        ],
        "state_on": [
            "{name} ist an.",
            "{name} ist eingeschaltet.",
            "{name} ist aktiv.",
        ],
        "state_off": [
            "{name} ist aus.",
            "{name} ist ausgeschaltet.",
            "{name} ist inaktiv.",
        ],
    },
    "fan": {
        "toggle": [
            "{name} ist jetzt {action}.",
            "Der Ventilator {name} ist {action}.",
            "Ich habe {name} {action}geschaltet.",
            "{name} läuft{action_suffix}.",
            "{name} ist {action}.",
        ],
        "state_on": [
            "{name} läuft.",
            "{name} ist an.",
            "{name} ist eingeschaltet.",
        ],
        "state_off": [
            "{name} läuft nicht.",
            "{name} ist aus.",
            "{name} ist ausgeschaltet.",
        ],
    },
    "vacuum": {
        "start": [
            "Staubsauger gestartet.",
            "Der Staubsauger läuft.",
            "Ich habe den Staubsauger gestartet.",
            "Staubsauger ist unterwegs.",
            "Der Staubsauger macht sich an die Arbeit.",
        ],
        "start_area": [
            "Staubsauger saugt jetzt {area}.",
            "Ich schicke den Staubsauger in {area}.",
            "Staubsauger reinigt {area}.",
            "{area} wird gesaugt.",
            "Der Staubsauger kümmert sich um {area}.",
        ],
        "state_on": [
            "Der Staubsauger läuft.",
            "Der Staubsauger ist unterwegs.",
            "Der Staubsauger saugt gerade.",
        ],
        "state_off": [
            "Der Staubsauger ist in der Station.",
            "Der Staubsauger läuft nicht.",
            "Der Staubsauger ist fertig.",
        ],
    },
    "climate": {
        "set_temperature": [
            "{name} ist auf {value}° eingestellt.",
            "Ich habe {name} auf {value}° gestellt.",
            "Temperatur von {name} ist {value}°.",
            "{name} heizt auf {value}°.",
            "Zieltemperatur für {name} ist {value}°.",
        ],
        "state": [
            "{name} ist auf {value}° eingestellt.",
            "{name} heizt auf {value}°.",
            "Temperatur in {name} ist {value}°.",
        ],
    },
    # Fallback for unknown domains
    "default": {
        "toggle": [
            "{name} ist jetzt {action}.",
            "Ich habe {name} {action}gemacht.",
            "{name} ist {action}.",
        ],
        "set": [
            "{name} ist auf {value} eingestellt.",
            "Ich habe {name} auf {value} gesetzt.",
            "{name} ist jetzt bei {value}.",
        ],
        "state": [
            "{name} ist {state}.",
        ],
    },
}


def get_domain_confirmation(
    domain: str,
    action: str,
    name: str = "",
    value: str = "",
    area: str = "",
    state: str = "",
) -> str:
    """Get a random confirmation message for a domain/action.
    
    Args:
        domain: Entity domain (light, cover, switch, etc.)
        action: Action type (toggle, open, close, brightness_set, etc.)
        name: Device/area name
        value: Value for brightness/position/temperature
        area: Area name for vacuum
        state: State for query responses (on/off)
        
    Returns:
        Formatted German confirmation message
    """
    # Get domain templates, fall back to default
    domain_templates = DOMAIN_RESPONSES.get(domain, DOMAIN_RESPONSES["default"])
    
    # Get action templates
    templates = domain_templates.get(action)
    if not templates:
        # Fall back to default domain
        templates = DOMAIN_RESPONSES["default"].get(action, ["{name} erledigt."])
    
    # Pick random template
    template = random.choice(templates)
    
    # Prepare action verb suffix (for "läuft" vs "läuft nicht")
    action_suffix = "" if state == "on" or action == "on" else " nicht"
    
    # Format with provided values
    try:
        return template.format(
            name=name,
            value=value,
            area=area,
            state=state,
            action=ACTION_VERBS.get(state, state),
            action_suffix=action_suffix,
        )
    except KeyError:
        return f"{name} erledigt."
