"""German language utilities for text processing and date handling.

Provides centralized German-specific logic for articles, weekdays, relative dates,
and yes/no response detection.
"""

import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Set, Tuple


# --- Articles and Prepositions ---

GERMAN_ARTICLES: Set[str] = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
}

GERMAN_PREPOSITIONS: Set[str] = {
    "im", "in", "auf", "unter", "über", "an", "am", "bei",
    "zum", "zur", "vom", "von", "für", "mit", "nach",
}


def nominative_to_accusative(phrase: str) -> str:
    """Convert nominative article to accusative case.
    
    German accusative rule: only masculine articles change (der → den).
    Neutral (das) and feminine/plural (die) stay the same.
    
    Args:
        phrase: Phrase starting with article, e.g. "der Rollladen"
        
    Returns:
        Phrase with accusative article, e.g. "den Rollladen"
        
    Examples:
        nominative_to_accusative("der Rollladen") -> "den Rollladen"
        nominative_to_accusative("das Licht") -> "das Licht"
        nominative_to_accusative("die Lampe") -> "die Lampe"
    """
    if not phrase:
        return phrase
    
    words = phrase.split()
    if not words:
        return phrase
    
    article = words[0].lower()
    if article == "der":
        # Masculine nominative → accusative
        words[0] = "den" if words[0].islower() else "Den"
    
    return " ".join(words)


def nominative_to_dative(phrase: str) -> str:
    """Convert nominative article to dative case.
    
    German dative: der → dem, das → dem, die → der (singular fem) / den (plural).
    Used after prepositions like 'von', 'mit', 'bei', etc.
    
    Args:
        phrase: Phrase starting with article, e.g. "das Licht"
        
    Returns:
        Phrase with dative article, e.g. "dem Licht"
        
    Examples:
        nominative_to_dative("das Licht") -> "dem Licht"  (neutral)
        nominative_to_dative("der Rollladen") -> "dem Rollladen"  (masculine)
        nominative_to_dative("die Lampe") -> "der Lampe"  (feminine singular)
    """
    if not phrase:
        return phrase
    
    words = phrase.split()
    if not words:
        return phrase
    
    article = words[0].lower()
    # Note: Can't distinguish feminine singular "die" from plural "die"
    # Assume singular for entity patterns
    dative_map = {
        "der": "dem",  # Masculine
        "das": "dem",  # Neutral
        "die": "der",  # Feminine singular (plural would be "den")
    }
    
    if article in dative_map:
        new_article = dative_map[article]
        words[0] = new_article if words[0].islower() else new_article.capitalize()
    
    return " ".join(words)


def capitalize_article_phrase(phrase: str) -> str:
    """Capitalize an article+noun phrase properly for German.
    
    Keeps article lowercase but capitalizes the noun.
    
    Args:
        phrase: e.g. "der rollladen" or "die rollläden"
        
    Returns:
        Properly capitalized phrase: "der Rollladen", "die Rollläden"
    """
    if not phrase:
        return phrase
    
    words = phrase.split()
    if len(words) < 2:
        return phrase
    
    # Article stays as-is, capitalize rest
    result = [words[0]]  # Keep article case
    for word in words[1:]:
        result.append(word.capitalize())
    
    return " ".join(result)


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


# --- Cache Normalization ---

def normalize_for_cache(text: str) -> Tuple[str, List]:
    """Normalize numeric values for semantic cache matching.
    
    Replaces variable numbers with canonical values for consistent cache lookup:
    - Percentages: "30%" → "50 Prozent"
    - Temperatures: "22 Grad" → "20 Grad"
    - Temporal delays: "in 3 Minuten" → "in Minuten" (number stripped)
    - Temporal times: "um 15:30 Uhr" → "um Uhr" (time stripped)
    - Durations: "für 5 Minuten" → "für Minuten" (number stripped)
    
    Uses duration_utils.py for consistent German duration parsing.
    
    Args:
        text: User input text to normalize
        
    Returns:
        Tuple of (normalized_text, extracted_values)
        
    Examples:
        normalize_for_cache("Licht auf 30%") → ("Licht auf 50 Prozent", [30])
        normalize_for_cache("in 5 Minuten aus") → ("in Minuten aus", [5])
    """
    from .duration_utils import parse_german_duration
    
    extracted: List = []

    def replace_percent(match):
        val = int(match.group(1))
        extracted.append(val)
        return "50 Prozent"

    def replace_temp(match):
        val = int(match.group(1))
        extracted.append(val)
        return "20 Grad"

    def replace_delay(match):
        # Use duration_utils for robust parsing
        full_match = match.group(0)  # e.g., "in 3 Minuten"
        duration_part = match.group(1) + " " + match.group(2)  # "3 Minuten"
        seconds = parse_german_duration(duration_part)
        extracted.append(seconds // 60 if seconds >= 60 else seconds)  # Store minutes or seconds
        # Return normalized form without number
        unit = match.group(2)
        return f"in {unit}"

    def replace_time(match):
        # Extract time: "um 15:30 Uhr" or "um 8 Uhr"
        time_str = match.group(1)
        extracted.append(time_str)
        return "um Uhr"

    def replace_duration(match):
        # "für X Minuten" → temporary control duration
        # Use duration_utils for robust parsing
        duration_part = match.group(1) + " " + match.group(2)  # "3 Minuten"
        seconds = parse_german_duration(duration_part)
        extracted.append(seconds // 60 if seconds >= 60 else seconds)
        # Return normalized form without number
        unit = match.group(2)
        return f"für {unit}"

    # Percentage patterns
    text_norm = re.sub(r"(\d+)\s*%", replace_percent, text)
    text_norm = re.sub(r"(\d+)\s*(prozent|Prozent)", replace_percent, text_norm)
    
    # Temperature patterns
    text_norm = re.sub(r"(\d+)\s*(grad|Grad)", replace_temp, text_norm)
    
    # Temporal delay patterns: "in 3 Minuten", "in einer Stunde", "in 10 Sekunden"
    # For DelayedControl - action AFTER delay
    text_norm = re.sub(
        r"\bin\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b",
        replace_delay, text_norm, flags=re.IGNORECASE
    )
    
    # Temporal time patterns: "um 15:30 Uhr", "um 8 Uhr"
    # For DelayedControl - action at specific time
    text_norm = re.sub(
        r"\bum\s+(\d{1,2}(?::\d{2})?)\s*Uhr\b",
        replace_time, text_norm, flags=re.IGNORECASE
    )
    
    # Duration patterns: "für 3 Minuten", "für eine Stunde"
    # For TemporaryControl - action NOW, revert after duration
    text_norm = re.sub(
        r"\bfür\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b",
        replace_duration, text_norm, flags=re.IGNORECASE
    )
    
    # Timer duration patterns: "auf 5 Minuten", "Timer auf 10 Minuten"
    # For HassTimerSet
    text_norm = re.sub(
        r"\bauf\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b",
        replace_duration, text_norm, flags=re.IGNORECASE
    )

    return text_norm, extracted
