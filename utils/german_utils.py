"""German language utilities for text processing and date handling.

Provides centralized German-specific logic for articles, weekdays, relative dates,
and yes/no response detection.
"""

import re
from datetime import date, datetime, timedelta
from typing import Optional, Set


# --- Articles and Prepositions ---

GERMAN_ARTICLES: Set[str] = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
}

GERMAN_PREPOSITIONS: Set[str] = {
    "im", "in", "auf", "unter", "über", "an", "am", "bei",
    "zum", "zur", "vom", "von", "für", "mit", "nach",
}


def remove_articles(text: str) -> str:
    """Remove German articles from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with articles removed
        
    Examples:
        remove_articles("den Keller") -> "Keller"
        remove_articles("die Küche") -> "Küche"
        remove_articles("das Bad") -> "Bad"
    """
    if not text:
        return ""
    
    words = text.split()
    filtered = [w for w in words if w.lower() not in GERMAN_ARTICLES]
    return " ".join(filtered)


def remove_prepositions(text: str) -> str:
    """Remove German prepositions from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with prepositions removed
        
    Examples:
        remove_prepositions("im Wohnzimmer") -> "Wohnzimmer"
        remove_prepositions("auf dem Tisch") -> "dem Tisch"
    """
    if not text:
        return ""
    
    words = text.split()
    filtered = [w for w in words if w.lower() not in GERMAN_PREPOSITIONS]
    return " ".join(filtered)


def remove_articles_and_prepositions(text: str) -> str:
    """Remove both articles and prepositions from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with articles and prepositions removed
        
    Example:
        remove_articles_and_prepositions("im den Keller") -> "Keller"
    """
    if not text:
        return ""
    
    words = text.split()
    stop_words = GERMAN_ARTICLES | GERMAN_PREPOSITIONS
    filtered = [w for w in words if w.lower() not in stop_words]
    return " ".join(filtered)


def canonicalize(text: str) -> str:
    """Canonicalize text for fuzzy matching.
    
    Performs:
    - Lowercase conversion
    - German umlaut normalization (ä→ae, ö→oe, ü→ue, ß→ss)
    - Punctuation removal
    - Whitespace normalization
    
    Args:
        text: Input text
        
    Returns:
        Canonicalized text for comparison
        
    Examples:
        canonicalize("Küche") -> "kueche"
        canonicalize("Gäste-Bad") -> "gaeste bad"
        canonicalize("  Büro  ") -> "buero"
    """
    if not text:
        return ""
    
    t = text.lower()
    t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    t = re.sub(r"[^\w\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --- Affirmative/Negative Detection ---

AFFIRMATIVE_WORDS: Set[str] = {
    "ja", "ok", "okay", "genau", "richtig", "passt", "korrekt",
    "stimmt", "gut", "jawohl", "jep", "jup", "sicher", "natürlich",
    "gerne", "bitte", "mach", "tu", "los",
}

NEGATIVE_WORDS: Set[str] = {
    "nein", "nicht", "abbrechen", "stop", "stopp", "falsch",
    "cancel", "weg", "vergiss", "lass", "ende", "beenden",
}


def is_affirmative(text: str) -> bool:
    """Check if text is an affirmative response.
    
    Args:
        text: User's response text
        
    Returns:
        True if response is affirmative
        
    Examples:
        is_affirmative("ja") -> True
        is_affirmative("ok, machen wir") -> True
        is_affirmative("nein danke") -> False
    """
    if not text:
        return False
    
    words = set(text.lower().split())
    return bool(words & AFFIRMATIVE_WORDS)


def is_negative(text: str) -> bool:
    """Check if text is a negative response.
    
    Args:
        text: User's response text
        
    Returns:
        True if response is negative
        
    Examples:
        is_negative("nein") -> True
        is_negative("abbrechen bitte") -> True
        is_negative("ja") -> False
    """
    if not text:
        return False
    
    words = set(text.lower().split())
    return bool(words & NEGATIVE_WORDS)


# --- Weekday Handling ---

WEEKDAYS_DE = {
    "montag": 0,
    "dienstag": 1,
    "mittwoch": 2,
    "donnerstag": 3,
    "freitag": 4,
    "samstag": 5,
    "sonntag": 6,
}

WEEKDAY_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def parse_weekday(text: str) -> Optional[int]:
    """Parse German weekday name to weekday number.
    
    Args:
        text: Text containing weekday name
        
    Returns:
        Weekday number (0=Monday, 6=Sunday) or None if not found
        
    Examples:
        parse_weekday("montag") -> 0
        parse_weekday("am Sonntag") -> 6
    """
    if not text:
        return None
    
    text_lower = text.lower()
    for day_name, day_num in WEEKDAYS_DE.items():
        if day_name in text_lower:
            return day_num
    return None


def get_next_weekday(weekday: int, from_date: Optional[date] = None) -> date:
    """Get next occurrence of a weekday.
    
    Args:
        weekday: Weekday number (0=Monday, 6=Sunday)
        from_date: Start date (default: today)
        
    Returns:
        Date of next occurrence (at least 1 day in future)
        
    Example:
        # If today is Wednesday (2)
        get_next_weekday(0)  # Next Monday
        get_next_weekday(2)  # Next Wednesday (7 days from now)
    """
    if from_date is None:
        from_date = date.today()
    
    days_ahead = weekday - from_date.weekday()
    if days_ahead <= 0:  # Target weekday is today or in the past
        days_ahead += 7
    
    return from_date + timedelta(days=days_ahead)


def get_weekday_name(weekday: int) -> str:
    """Get German weekday name from number.
    
    Args:
        weekday: Weekday number (0=Monday, 6=Sunday)
        
    Returns:
        German weekday name
    """
    return WEEKDAY_NAMES[weekday % 7]


# --- Relative Date Handling ---

# Relative date terms: (term, days_offset)
# Ordered by length (longest first) to avoid partial matches
RELATIVE_DATES = [
    ("übermorgen", 2),
    ("morgen", 1),
    ("heute", 0),
]


def parse_relative_date(text: str, from_date: Optional[date] = None) -> Optional[date]:
    """Parse German relative date expressions.
    
    Args:
        text: Text containing date expression
        from_date: Reference date (default: today)
        
    Returns:
        Resolved date or None if not parseable
        
    Supported patterns:
        - heute, morgen, übermorgen
        - in X Tagen
        - X Tage
        - nächsten Montag, am Dienstag
        
    Examples:
        parse_relative_date("morgen") -> tomorrow's date
        parse_relative_date("in 5 Tagen") -> 5 days from now
        parse_relative_date("nächsten Montag") -> next Monday
    """
    if not text:
        return None
    
    if from_date is None:
        from_date = date.today()
    
    text_lower = text.lower().strip()
    
    # Check relative day terms (heute, morgen, übermorgen)
    for term, days_offset in RELATIVE_DATES:
        if term in text_lower:
            return from_date + timedelta(days=days_offset)
    
    # Check "in X Tagen" pattern
    match = re.search(r'in\s+(\d+)\s+tag', text_lower)
    if match:
        days = int(match.group(1))
        return from_date + timedelta(days=days)
    
    # Check "X Tage" pattern (without "in")
    match = re.match(r'(\d+)\s+tag', text_lower)
    if match:
        days = int(match.group(1))
        return from_date + timedelta(days=days)
    
    # Check weekday patterns ("nächsten Montag", "am Dienstag")
    weekday = parse_weekday(text)
    if weekday is not None:
        return get_next_weekday(weekday, from_date)
    
    return None


def resolve_relative_date_str(value: str, from_date: Optional[date] = None) -> str:
    """Resolve relative date string to YYYY-MM-DD format.
    
    Args:
        value: Date string (may be relative or already formatted)
        from_date: Reference date (default: today)
        
    Returns:
        Date in YYYY-MM-DD format, or original value if not parseable
        
    Examples:
        resolve_relative_date_str("morgen") -> "2024-12-15"
        resolve_relative_date_str("2024-12-15") -> "2024-12-15" (unchanged)
    """
    if not value:
        return value
    
    # Already in correct format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
        return value
    
    resolved = parse_relative_date(value, from_date)
    if resolved:
        return resolved.strftime("%Y-%m-%d")
    
    return value


# --- Date/Time Formatting ---

def format_date_german(d: date) -> str:
    """Format date in German style.
    
    Args:
        d: Date to format
        
    Returns:
        German format: "DD.MM.YYYY"
    """
    return d.strftime("%d.%m.%Y")


def format_datetime_german(dt: datetime) -> str:
    """Format datetime in German style.
    
    Args:
        dt: Datetime to format
        
    Returns:
        German format: "DD.MM.YYYY um HH:MM Uhr"
    """
    return dt.strftime("%d.%m.%Y um %H:%M Uhr")
