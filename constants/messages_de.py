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

from typing import Dict, Optional


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
    
    # Disambiguation messages
    "multiple_matches": "Ich habe mehrere passende Geräte gefunden:",
    "please_specify": "Bitte sag mir genauer, welches du meinst.",
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
