"""German language utilities for text processing and date handling.

Provides centralized German-specific logic for articles, weekdays, relative dates,
and yes/no response detection.
"""

import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple


try:
    from ..constants.messages_de import (
        IMPLICIT_PHRASES,
        EXIT_COMMANDS,
        AFFIRMATIVE_WORDS,
        NEGATIVE_WORDS,
        DOMAIN_DESCRIPTIONS,
        LEARNING_OFFER_PROMPTS as LEARNING_CONFIRMATION_PROMPTS,
    )
except (ImportError, ValueError):
    from constants.messages_de import (
        IMPLICIT_PHRASES,
        EXIT_COMMANDS,
        AFFIRMATIVE_WORDS,
        NEGATIVE_WORDS,
        DOMAIN_DESCRIPTIONS,
        LEARNING_OFFER_PROMPTS as LEARNING_CONFIRMATION_PROMPTS,
    )
try:
    from ..constants.entity_keywords import (
        FRACTION_VALUES as FRACTION_INT_MAPPINGS,
        STATE_TRANSLATIONS as HA_STATE_TRANSLATIONS,
    )
except (ImportError, ValueError):
    from constants.entity_keywords import (
        FRACTION_VALUES as FRACTION_INT_MAPPINGS,
        STATE_TRANSLATIONS as HA_STATE_TRANSLATIONS,
    )
try:
    from ..constants.area_keywords import (
        AREA_ALIASES,
        AREA_PREPOSITIONS,
        AREA_INDICATORS,
        FLOOR_KEYWORDS,
        LOCATION_INDICATORS,
    )
except (ImportError, ValueError):
    from constants.area_keywords import (
        AREA_ALIASES,
        AREA_PREPOSITIONS,
        AREA_INDICATORS,
        FLOOR_KEYWORDS,
        LOCATION_INDICATORS,
    )

# Internal language base constants
GERMAN_ARTICLES: Set[str] = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
}

GERMAN_PREPOSITIONS: Set[str] = {
    "im", "in", "auf", "unter", "über", "am", "bei", # Removed "an" as preposition to protect it as verbal particle
    "zum", "zur", "vom", "von", "für", "mit", "nach",
}

# Mapping for response-facing state translations
STATE_TRANSLATIONS: Dict[str, str] = {
    "closing": "schließt", "opening": "öffnet", "buffering": "lädt",
    "playing": "spielt", "paused": "pausiert", "idle": "inaktiv",
    "off": "aus", "on": "an", "open": "offen", "closed": "geschlossen",
    "unavailable": "nicht verfügbar",
}

COMPOUND_SEPARATOR: str = " und "


def normalize_umlauts(text: str) -> str:
    """Normalize German umlauts and ß to ASCII equivalents."""
    return text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")


def remove_articles_and_prepositions(text: str) -> str:
    """Remove German articles and prepositions from text."""
    if not text:
        return ""
    return " ".join(
        w for w in text.split()
        if w.lower() not in GERMAN_ARTICLES and w.lower() not in GERMAN_PREPOSITIONS
    )


def map_area_alias(text: str) -> str:
    """Robust room alias mapping."""
    if not text: return text

    words = text.split()
    mapped = []
    for word in words:
        clean = word.lower().strip(",.!?:")
        if clean in AREA_ALIASES:
            target = AREA_ALIASES[clean]
            mapped.append(target if word[0].isupper() else target.lower())
        else:
            canon = normalize_umlauts(clean)
            if canon in AREA_ALIASES:
                target = AREA_ALIASES[canon]
                mapped.append(target if word[0].isupper() else target.lower())
            else:
                mapped.append(word)
    return " ".join(mapped)


def get_prepositional_area(area_name: str) -> str:
    if not area_name: return area_name
    area_lower = area_name.lower()
    if area_lower in AREA_PREPOSITIONS:
        return f"{AREA_PREPOSITIONS[area_lower]} {area_name}"
    if area_lower.endswith(("zimmer", "bad", "flur", "garten", "garage", "dachboden", "büro", "keller")):
        return f"im {area_name}"
    if area_lower.endswith(("etage", "ebene")):
        return f"auf der {area_name}"
    return f"in {area_name}"


def canonicalize(text: str) -> str:
    if not text: return ""
    import unicodedata
    t = unicodedata.normalize('NFC', text) # Preserving casing as requested
    t = normalize_umlauts(t)
    t = re.sub(r"[^\w\s%°]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


WEEKDAYS_DE = {"montag": 0, "dienstag": 1, "mittwoch": 2, "donnerstag": 3, "freitag": 4, "samstag": 5, "sonntag": 6}
WEEKDAY_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]

def parse_weekday(text: str) -> Optional[int]:
    if not text: return None
    t = text.lower()
    for day, num in WEEKDAYS_DE.items():
        if day in t: return num
    return None

def get_next_weekday(weekday: int, from_date: Optional[date] = None) -> date:
    if from_date is None: from_date = date.today()
    days = weekday - from_date.weekday()
    if days <= 0: days += 7
    return from_date + timedelta(days=days)

def get_weekday_name(weekday: int) -> str: return WEEKDAY_NAMES[weekday % 7]

RELATIVE_DATES = [("übermorgen", 2), ("morgen", 1), ("heute", 0)]

def parse_relative_date(text: str, from_date: Optional[date] = None) -> Optional[date]:
    if not text: return None
    if from_date is None: from_date = date.today()
    t = text.lower().strip()
    for term, offset in RELATIVE_DATES:
        if term in t: return from_date + timedelta(days=offset)
    m = re.search(r'in\s+(\d+)\s+tag', t)
    if m: return from_date + timedelta(days=int(m.group(1)))
    m = re.match(r'(\d+)\s+tag', t)
    if m: return from_date + timedelta(days=int(m.group(1)))
    wd = parse_weekday(text)
    return get_next_weekday(wd, from_date) if wd is not None else None

def resolve_relative_date_str(value: str, from_date: Optional[date] = None) -> str:
    if not value or re.match(r'^\d{4}-\d{2}-\d{2}$', value): return value
    resolved = parse_relative_date(value, from_date)
    return resolved.strftime("%Y-%m-%d") if resolved else value

def format_date_german(d: date) -> str: return d.strftime("%d.%m.%Y")
def format_datetime_german(dt: datetime) -> str: return dt.strftime("%d.%m.%Y um %H:%M Uhr")


def is_affirmative(text: str) -> bool:
    """Check if text is a German affirmative response."""
    if not text: return False
    t = canonicalize(text)
    return any(w in t.split() for w in AFFIRMATIVE_WORDS)


def is_negative(text: str) -> bool:
    """Check if text is a German negative response."""
    if not text: return False
    t = canonicalize(text)
    return any(w in t.split() for w in NEGATIVE_WORDS)


# FILLER_WORDS: Only truly meaningless words that add no semantic value.
# Articles and prepositions are INTENTIONALLY EXCLUDED — they carry
# grammatical and spatial meaning ("im Keller" vs "auf dem Balkon").
FILLER_WORDS: Set[str] = {"bitte", "mal", "gerne", "doch", "kannst", "könntest", "würdest"}


def strip_filler_words(text: str) -> str:
    """Strip only meaningless filler words from text.

    Preserves articles, prepositions, and all intent-critical particles.
    The embedding model is trained on proper German and handles grammar natively.
    """
    tokens = text.split()
    return " ".join([w for w in tokens if w.lower() not in FILLER_WORDS]).strip()





def normalize_for_cache(text: str) -> Tuple[str, List]:
    """Normalize text for semantic cache lookup.

    Only normalizes numbers/times to centroids (Principle 7) and strips filler words.
    Preserves German grammar (articles, prepositions) because:
    - "im Keller" vs "auf dem Balkon" carries spatial meaning (Principle 2)
    - "der Rollladen" vs "die Rollläden" carries plurality (Principle 1)
    - "an" vs "aus" carries intent (Principle 3)
    The embedding model is trained on proper German sentences.
    """
    if not text: return "", []
    if " und " in text or "," in text: return "[MULTIPLE_COMMANDS_ESCALATION]", []



    text = map_area_alias(text)
    text_norm = canonicalize(text)
    from .duration_utils import parse_german_duration
    extracted: List = []

    # 1. Number Centroids (Principle 7) — normalize values that can't
    #    be meaningfully represented in vector space
    def repl_pct(m):
        extracted.append(int(m.group(1)))
        return "50 prozent"
    def repl_temp(m):
        extracted.append(int(m.group(1)))
        return "21 grad"

    # \b doesn't work after % because % is not a word character.
    # Use (?:\s|$) as boundary instead.
    text_norm = re.sub(r"(\d+)\s*(?:%|prozent)(?:\b|\s|$)", repl_pct, text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"(\d+)\s*(?:\u00b0|grad)\b", repl_temp, text_norm, flags=re.IGNORECASE)

    # 2. Time Centroids — normalize all temporal expressions to standard form
    #    so "in 5 Minuten" and "in 2 Stunden" match the same cache entry
    def repl_delay_in(m):
        sec = parse_german_duration(m.group(1) + " " + m.group(2))
        extracted.append(sec // 60 if sec >= 60 else sec)
        return "in 10 minuten"

    def repl_delay_fuer(m):
        sec = parse_german_duration(m.group(1) + " " + m.group(2))
        extracted.append(sec // 60 if sec >= 60 else sec)
        return "fuer 10 minuten"

    def repl_delay_auf(m):
        sec = parse_german_duration(m.group(1) + " " + m.group(2))
        extracted.append(sec // 60 if sec >= 60 else sec)
        return "auf 10 minuten"

    def repl_clock(m):
        extracted.append(m.group(1))
        return "um 10 uhr"

    # "in 5 Minuten", "in 2 Stunden", "in 30 Sekunden"
    text_norm = re.sub(r"\bin\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", repl_delay_in, text_norm, flags=re.IGNORECASE)
    # "für 5 Minuten", "für 2 Stunden"
    text_norm = re.sub(r"\bfuer\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", repl_delay_fuer, text_norm, flags=re.IGNORECASE)
    # "auf 5 Minuten" (timer duration)
    text_norm = re.sub(r"\bauf\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", repl_delay_auf, text_norm, flags=re.IGNORECASE)
    # "um 15:30 Uhr" → after canonicalize, ":" is stripped → "um 15 30 uhr"
    # Also handle "um 8 uhr" (no minutes)
    text_norm = re.sub(r"\bum\s+(\d{1,2})(?:\s+\d{2})?\s*uhr\b", repl_clock, text_norm, flags=re.IGNORECASE)

    # 3. Fraction normalization ("zur Hälfte" → "50 Prozent")
    # FRACTION_VALUES keys may contain umlauts ("hälfte") but canonicalize()
    # has already converted them ("haelfte"), so we canonicalize the keys too.
    try:
        from ..constants.entity_keywords import FRACTION_VALUES
    except (ImportError, ValueError):
        from constants.entity_keywords import FRACTION_VALUES
    for fraction_word, fraction_val in FRACTION_VALUES.items():
        canon_word = canonicalize(fraction_word)
        pattern = r"\b" + re.escape(canon_word) + r"\b"
        if re.search(pattern, text_norm, flags=re.IGNORECASE):
            extracted.append(fraction_val)
            text_norm = re.sub(pattern, "50 prozent", text_norm, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text_norm).strip(), extracted


def extract_delay(text: str) -> Optional[str]:
    m = re.search(r"\bin\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b", text, re.IGNORECASE)
    if m: return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\bum\s+(\d{1,2}(?::\d{2})?)\s*Uhr\b", text, re.IGNORECASE)
    return f"{m.group(1)} Uhr" if m else None

def extract_duration(text: str) -> Optional[str]:
    m = re.search(r"\bfür\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b", text, re.IGNORECASE)
    return f"{m.group(1)} {m.group(2)}" if m else None

def extract_timer_duration(text: str) -> Optional[str]:
    for p in [r"\bfür\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b", 
              r"\bauf\s+(\d+|eine[rn]?)\s+(Minuten?|Stunden?|Sekunden?)\b",
              r"(\d+)\s*(Minuten?|Stunden?|Sekunden?)\s+(?:timer|wecker)\b"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return f"{m.group(1)} {m.group(2)}"
    return None
