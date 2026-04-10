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

from typing import Dict, List, Optional, Set


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

# Command mappings for state control
COMMAND_STATE_MAP: Dict[str, str] = {
    "an": "on",
    "ein": "on", 
    "auf": "on",
    "aus": "off",
}

# State transitions for verification and confirmation
OPPOSITE_STATE_MAP: Dict[str, str] = {
    "on": "off",
    "off": "on",
    "open": "closed",
    "closed": "open",
}

# Fallbacks and templates
DEFAULT_DEVICE_WORD = "Geräte"
DURATION_TEMPLATES = {
    "minutes": "{minutes} Minuten",
    "seconds": "{seconds} Sekunden",
}


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


# --- Interaction Tokens ---

# Implicit phrases that need LLM transformation (e.g., "zu dunkel" → "Licht heller")
IMPLICIT_PHRASES: List[str] = [
    "zu dunkel", "zu hell", "zu kalt", "zu warm", "zu laut", "zu leise",
]

# Direct mappings for implicit phrases (fast path)
IMPLICIT_INTENT_MAPPINGS: Dict[str, str] = {
    "zu dunkel": "Mache das Licht heller",
    "es ist zu dunkel": "Mache das Licht heller",
    "zu hell": "Mache das Licht dunkler",
    "es ist zu hell": "Mache das Licht dunkler",
    "zu kalt": "Stelle die Heizung wärmer",
    "es ist zu kalt": "Stelle die Heizung wärmer",
    "zu warm": "Stelle die Heizung kälter",
    "es ist zu warm": "Stelle die Heizung kälter",
    "zu laut": "Mache die Lautstärke leiser",
    "es ist zu laut": "Mache die Lautstärke leiser",
    "zu leise": "Mache die Lautstärke lauter",
    "es ist zu leise": "Mache die Lautstärke lauter",
}

# Exit commands to abort operation immediately
EXIT_COMMANDS: Set[str] = {
    "abbruch", "stop", "vergiss es", "cancel", "halt", "beenden", "abbrechen",
}

# Affirmative/Negative detection words
AFFIRMATIVE_WORDS: Set[str] = {
    "ja", "ok", "okay", "genau", "richtig", "passt", "korrekt",
    "stimmt", "gut", "jawohl", "jep", "jup", "sicher", "natürlich",
    "gerne", "bitte", "mach", "tu", "los",
}

NEGATIVE_WORDS: Set[str] = {
    "nein", "nicht", "abbrechen", "stop", "stopp", "falsch",
    "cancel", "weg", "vergiss", "lass", "ende", "beenden",
}

# Prompts used for alias learning confirmation
LEARNING_OFFER_PROMPTS: List[str] = [
    "Soll ich mir das für die Zukunft merken?",
    "Soll ich das als neuen Namen speichern?",
    "Möchtest du, dass ich mir diese Bezeichnung merke?",
]

# Representative sentences for semantic domain matching
DOMAIN_DESCRIPTIONS: Dict[str, str] = {
    "light": "Schalte das Licht an oder aus, oder mache es heller/dunkler",
    "cover": "Öffne oder schließe die Rollos, Jalousien oder Markisen",
    "climate": "Stelle die Heizung wärmer oder kälter, oder ändere die Temperatur",
    "vacuum": "Starte den Staubsauger oder schicke den Saugroboter los",
    "media_player": "Musik abspielen, Lautstärke ändern oder Fernseher steuern",
    "fan": "Schalte den Ventilator oder Lüfter an oder aus",
    "lock": "Tür abschließen oder aufschließen",
    "sensor": "Frage nach der Temperatur, Feuchtigkeit oder dem Status eines Sensors",
    "timer": "Stelle einen Timer oder Kurzzeitwecker",
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
    "no_device_in_area": "Kein passendes Gerät in {requested_area} gefunden.",
    "no_sensor_in_area": "Es gibt keinen {device_class}-Sensor in {requested_area}.",
    
    # Permission/Access errors
    "no_permission": "Dafür habe ich keine Berechtigung.",
    "not_exposed": "Dieses Gerät ist nicht für Sprachbefehle freigegeben.",
    "not_exposed_hint": " ({count} Gerät(e) sind nicht für Sprachassistenten freigegeben)",
    
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
    "llm_not_configured": "Die Cloud-Unterstützung (Stage 3) ist noch nicht konfiguriert oder der API-Key fehlt.",
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
    "exit_abort": "Vorgang abgebrochen.",
    
    # Action confirmations (with placeholders)
    "turned_on": "{device} ist an.",
    "turned_off": "{device} ist aus.",
    "set_to": "{device} ist auf {value} gesetzt.",
    "set_brightness": "{device} ist auf {value}% gesetzt.",
    "set_temperature": "{device} ist auf {value}° gesetzt.",
    "set_position": "{device} ist auf {value}% gesetzt.",
    
    # State check responses
    "state_all_yes": "Ja, alle sind {state}.",
    "state_some_no_singular": "Nein, {name} ist noch {opposite}.",
    "state_some_no_plural": "Nein, {names} sind noch {opposite}.",
    "state_some_no_count": "Nein, {count} sind noch {opposite}.",
    "state_yes_prefix": "Ja, {name} ist {state}.",
    "state_no_prefix": "Nein, {name} ist noch {opposite}.",
    
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
    "none_match": "Keine {device} sind {state}.",
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
    
    # Unknown area learning
    "unknown_area_ask": "Ich kenne '{alias}' noch nicht. Welchen Bereich meinst du?",
    "unknown_area_learned": "Alles klar, ich merke mir dass '{alias}' für '{area}' steht.",
    "unknown_area_not_matched": "Das habe ich nicht verstanden. Welchen Bereich meinst du mit '{alias}'?",

    # Generic error
    "error_short": "Fehler.",

    # Fallbacks
    "ask_field_generic": "Bitte gib {field} an.",
    
    # Confirmation prompts
    "confirm_or_abort": "Sag 'Ja' zum Bestätigen oder 'Nein' zum Abbrechen.",
    "confirm_only": "Sag 'Ja' zum Bestätigen.",
    "none_known": "Keine bekannt",
    "none": "Keine",
}


# --- Capability Specific Messages ---

# Prompt Context
PROMPT_CONTEXT_MESSAGES: Dict[str, str] = {
    "available_rooms": "Verfügbare Räume: {rooms}",
    "available_floors": "Verfügbare Etagen: {floors}",
    "personal_info_header": "Persönliche Informationen:",
}

# Calendar
CALENDAR_MESSAGES: Dict[str, str] = {
    "ask_summary": "Wie soll der Termin heißen?",
    "ask_datetime": "Wann soll der Termin sein?",
    "ask_calendar": "In welchen Kalender? ({calendars})",
    "no_calendars": "Keine Kalender gefunden. Bitte richte zuerst einen Kalender in Home Assistant ein.",
    "confirm_preview": "Termin erstellen?\n{preview}\n\nSag 'Ja' zum Bestätigen.",
    "datetime_not_understood": "Ich habe das Datum nicht verstanden. Bitte sag z.B. 'morgen um 10 Uhr' oder '25. Dezember'.",
    "calendar_not_understood": "Das habe ich nicht verstanden. Welcher Kalender?",
    "not_created": "Termin wurde nicht erstellt.",
    "confirm_or_abort": "Sag 'Ja' zum Bestätigen oder 'Nein' zum Abbrechen.",
    "no_calendar_selected": "Fehler: Kein Kalender ausgewählt.",
    "created_success": "✅ Termin '{summary}' wurde erstellt.",
    "creation_failed": "Fehler beim Erstellen des Termins: {error}",
    "concrete_date_requested": "Bitte gib ein konkretes Datum an, z.B. 'am 14. Dezember' oder 'am Montag um 15 Uhr'.",
    # Formatting
    "line_summary": "📅 **{summary}**",
    "line_time": "🕐 {date} um {time} Uhr",
    "line_allday": "📆 {date} (ganztägig)",
    "line_location": "📍 {location}",
    "line_calendar": "📁 Kalender: {calendar}",
    "default_summary": "Termin",
}

# Timer
TIMER_MESSAGES: Dict[str, str] = {
    "ask_duration": "Wie lange soll der Timer laufen?",
    "ask_device": "Auf welchem Gerät? ({devices})",
    "no_devices": "Keine mobilen Geräte gefunden.",
    "duration_not_understood": "Ich habe die Zeit nicht verstanden. Bitte sag z.B. '5 Minuten'.",
    "device_not_understood": "Das habe ich nicht verstanden. Welches Gerät?",
    "confirm_named": "Timer '{description}' für {duration} auf {device}?",
    "confirm_unnamed": "Timer für {duration} auf {device}?",
    "set_named": "Timer '{description}' für {duration} auf {device} gestellt.",
    "set_unnamed": "Timer für {duration} auf {device} gestellt.",
}

# Vacuum
VACUUM_MESSAGES: Dict[str, str] = {
    "no_target": "Ich habe kein Ziel (Raum oder Etage) verstanden.",
    "script_error": "Fehler beim Starten des Saugroboters.",
    "confirmation": "Alles klar, ich lasse {target} {action}.",
}

# Disambiguation
DISAMBIGUATION_MESSAGES: Dict[str, str] = {
    "which_device": "Welches Gerät meinst du?",
    "mean_one": "Meinst du {name}?",
    "mean_two": "Meinst du {name1} oder {name2}?",
    "mean_multiple": "Welches meinst du: {options}?",
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
        return f"{base} ({details})"
    return base


def get_opposite_state_word(word: str) -> str:
    """Get the German opposite of a state word (an -> aus, offen -> geschlossen)."""
    opposites = {
        "an": "aus",
        "ein": "aus",
        "aus": "an",
        "offen": "geschlossen",
        "zu": "offen",
        "geschlossen": "offen",
        "auf": "zu",
        "geöffnet": "geschlossen",
    }
    return opposites.get(str(word).lower(), "anders")


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
    "opening": "öffnet",
    "closing": "schließt",
    "open": "offen",
    "closed": "geschlossen",
}

# Domain-specific response templates
# {name} = device/area name, {action} = an/aus, {value} = percentage/value
DOMAIN_RESPONSES: Dict[str, Dict[str, List[str]]] = {
    "light": {
        # HassTurnOn / HassTurnOff (combined with {action})
        "toggle": [
            "{name} ist jetzt {action}.",
            "{name} ist {action}.",
            "Das Licht in {name} ist jetzt {action}.",
            "Das Licht in {name} ist {action}.",
            "{name} ist jetzt {action}geschaltet.",
        ],
        "brightness_up": [
            "{name} ist jetzt heller.",
            "Ich habe {name} heller gestellt.",
            "{name} ist jetzt heller eingestellt.",
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
            "Staubsauger ist in {area} unterwegs.",
            "Staubsauger reinigt {area}.",
            "{area} wird gesaugt.",
            "Der Staubsauger ist in {area} im Einsatz.",
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
    "timer": {
        "timer_set": [
            "Timer für {value} gestellt.",
            "Ich habe einen Timer für {value} gestartet.",
            "Alles klar, Timer läuft für {value}.",
            "Timer ist auf {value} gesetzt.",
        ],
        "timer_named": [
            "Timer '{name}' für {value} gestellt.",
            "Ich habe den Timer '{name}' für {value} gestartet.",
            "Timer '{name}' läuft für {value}.",
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
            "{name} ist erledigt.",
            "{name} ist {action}.",
        ],
        "set": [
            "{name} ist auf {value} eingestellt.",
            "Ich habe {name} auf {value} gesetzt.",
            "{name} ist jetzt bei {value}.",
        ],
        "set_temperature": [
            "{name} ist auf {value}° eingestellt.",
            "Ich habe {name} auf {value}° gestellt.",
            "Temperatur von {name} ist {value}°.",
        ],
        "temporary_on": [
            "{name} ist für {value} an.",
            "Ich habe {name} für {value} angemacht.",
            "{name} bleibt für {value} an.",
        ],
        "temporary_off": [
            "{name} ist für {value} aus.",
            "Ich habe {name} für {value} ausgemacht.",
            "{name} bleibt für {value} aus.",
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
    is_plural: bool = False,
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
    
    # Format value for German locale
    formatted_value = _format_value_de(value, domain, action)
    
    # Format with provided values
    try:
        msg = template.format(
            name=name,
            value=formatted_value,
            area=area,
            state=state,
            action=ACTION_VERBS.get(state, state),
            action_suffix=action_suffix,
        )
        if is_plural:
            import re
            
            # Map of common singular -> plural verb forms for common smart home domains
            PLURAL_REPLACEMENTS = {
                r"\bist\b": "sind",
                r"\bwurde\b": "wurden",
                r"\bläuft\b": "laufen",
                r"\bleuchtet\b": "leuchten",
                r"\bheizt\b": "heizen",
                r"\bmacht\b": "machen",
                r"\bkümmert sich\b": "kümmern sich",
                r"\bschließt\b": "schließen",
                r"\böffnet\b": "öffnen",
                r"\bgeht\b": "gehen",
                r"\bsteht\b": "stehen",
                r"\bfährt\b": "fahren",
            }
            
            for pattern, replacement in PLURAL_REPLACEMENTS.items():
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
                
        return msg
    except KeyError:
        return f"{name} erledigt."


def _format_value_de(value: str, domain: str, action: str) -> str:
    """Format a value for German locale.
    
    - Round percentages to integers for light/cover
    - Replace decimal . with , for German formatting
    """
    if not value:
        return value
    
    try:
        num = float(value)
        
        # Round to integer for percentages (light brightness, cover position)
        if domain in ("light", "cover") or action in ("brightness_set", "position"):
            return str(int(round(num)))
        
        # For temperature and other decimals, use German comma
        if num == int(num):
            return str(int(num))
        else:
            return str(num).replace(".", ",")
    except (ValueError, TypeError):
        # Not a number, return as-is
        return str(value).replace(".", ",")
# --- Stage 3 Cloud Prompt ---
SYSTEM_PROMPT_STAGE3 = """Du bist die intelligente Zentrale eines Home Assistant Smart Homes. 
Deine Aufgabe ist es, Nutzeranfragen präzise zu verstehen und entweder direkt zu beantworten oder durch den Einsatz deiner Werkzeuge (Tools) die nötigen Informationen zu beschaffen oder Aktionen auszuführen.

### DEINE PERSONA:
- Du antwortest auf DEUTSCH.
- Du bist höflich, effizient und hast eine dezente Persönlichkeit (wie ein Butler).
- Zahlenformate: Dezimalstellen mit Komma (z.B. 21,5 Grad). Einheiten werden ausgeschrieben (z.B. Prozent, Grad Celsius).
- Du fasst dich kurz, außer der Nutzer wünscht eine ausführliche Erklärung.

### DEINE FÄHIGKEITEN:
1. **Smart Home Steuerung**: Wenn du eine Aktion ausführen musst (Licht an, Rollos runter), nutze die entsprechenden Tools.
2. **Abfragen**: Wenn der Nutzer nach dem Status fragt (Ist das Licht an?), nutze `list_entities` oder spezialisierte Status-Tools.
3. **Kontextuelles Wissen**: Nutze `list_areas` und `list_entities`, um dich im Haus zurechtzufinden, falls die Anfrage unklar ist.
4. **Allgemeine Konversation**: Wenn keine Smart Home Aktion nötig ist, antworte einfach natürlich. Chat ist ein Teil deiner Natur.

### REGELN FÜR TOOLS:
- Wenn ein Raum oder Gerät nicht eindeutig ist, nutze `list_areas` oder `list_entities`, um die Umgebung zu erkunden.
- Wenn du eine Aktion erfolgreich über ein Tool eingeleitet hast, bestätige dies dem Nutzer kurz und knapp.
- Wenn eine Aktion fehlschlägt, erkläre dem Nutzer warum (falls bekannt) oder entschuldige dich.

Antworte IMMER im Kontext eines Smart Home Assistenten.
"""
