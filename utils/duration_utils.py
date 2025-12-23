"""Duration parsing and formatting utilities for German time expressions.

Provides centralized duration handling for timers, temporary controls, and calendar events.
"""

import re
from typing import Any, Optional, Tuple


def parse_german_duration(text: Any) -> int:
    """Parse German duration text to seconds.
    
    Supports:
        - "5 Minuten" -> 300
        - "2 Stunden" -> 7200
        - "30 Sekunden" -> 30
        - "1,5 Stunden" -> 5400 (decimal with comma)
        - "1.5 Stunden" -> 5400 (decimal with period)
        - "2 Stunden 30 Minuten" -> 9000 (combined)
        - Integer input -> assumed as seconds
        - Plain number string -> assumed as minutes (e.g., "5" -> 300)
    
    Args:
        text: Duration text or integer
        
    Returns:
        Duration in seconds, or 0 if not parseable
        
    Examples:
        parse_german_duration("5 Minuten") -> 300
        parse_german_duration("1,5 Stunden") -> 5400
        parse_german_duration(120) -> 120
    """
    if not text:
        return 0
    
    # If already an integer, return as-is
    if isinstance(text, int):
        return text
    
    # Try parsing as plain integer (already in seconds)
    try:
        return int(text)
    except (ValueError, TypeError):
        pass
    
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower().strip()
    total = 0
    
    # First, normalize German word numbers to digits
    # "einer Stunde" → "1 Stunde", "eine Minute" → "1 Minute"
    word_to_num = {
        "eine": "1", "einer": "1", "einem": "1", "eins": "1",
        "zwei": "2", "drei": "3", "vier": "4", "fünf": "5",
        "sechs": "6", "sieben": "7", "acht": "8", "neun": "9", "zehn": "10",
        "halb": "0.5", "halbe": "0.5", "halben": "0.5",
    }
    for word, num in word_to_num.items():
        text_lower = re.sub(rf'\b{word}\b', num, text_lower)
    
    # Match hours (supports decimals like "1,5 Stunden" or "1.5 Stunden" or "0.5 Stunden")
    hours_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:stunden?|std|h)\b', text_lower)
    if hours_match:
        hours = float(hours_match.group(1).replace(',', '.'))
        total += int(hours * 3600)
    
    # Match minutes
    minutes_match = re.search(r'(\d+)\s*(?:minuten?|min|m)\b', text_lower)
    if minutes_match:
        total += int(minutes_match.group(1)) * 60
    
    # Match seconds
    seconds_match = re.search(r'(\d+)\s*(?:sekunden?|sec|s)\b', text_lower)
    if seconds_match:
        total += int(seconds_match.group(1))
    
    # If no unit matched but it's a plain number, assume minutes
    if total == 0 and text_lower.isdigit():
        return int(text_lower) * 60
    
    return total


def parse_duration_to_minutes(text: Any) -> Optional[int]:
    """Parse German duration text to minutes.
    
    Convenience wrapper for calendar/event durations that work in minutes.
    
    Args:
        text: Duration text
        
    Returns:
        Duration in minutes, or None if not parseable
        
    Examples:
        parse_duration_to_minutes("3 Stunden") -> 180
        parse_duration_to_minutes("30 Minuten") -> 30
        parse_duration_to_minutes("1,5 Stunden") -> 90
    """
    seconds = parse_german_duration(text)
    if seconds > 0:
        return seconds // 60
    return None


def parse_duration_to_components(text: Any) -> Tuple[int, int, int]:
    """Parse duration to (hours, minutes, seconds) tuple.
    
    Args:
        text: Duration text
        
    Returns:
        Tuple of (hours, minutes, seconds)
        
    Example:
        parse_duration_to_components("2 Stunden 30 Minuten") -> (2, 30, 0)
    """
    total_seconds = parse_german_duration(text)
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    seconds = remaining % 60
    return (hours, minutes, seconds)


def format_duration_german(seconds: int, short: bool = False) -> str:
    """Format seconds to German duration text.
    
    Args:
        seconds: Duration in seconds
        short: Use short format (e.g., "1 Std 30 Min" vs "1 Stunde 30 Minuten")
        
    Returns:
        Human-readable German duration string
        
    Examples:
        format_duration_german(300) -> "5 Minuten"
        format_duration_german(3665) -> "1 Stunde 1 Minute 5 Sekunden"
        format_duration_german(3665, short=True) -> "1 Std 1 Min 5 Sek"
    """
    if seconds <= 0:
        return "0 Sekunden" if not short else "0 Sek"
    
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    secs = remaining % 60
    
    parts = []
    
    if hours > 0:
        if short:
            parts.append(f"{hours} Std")
        else:
            parts.append(f"{hours} Stunde" if hours == 1 else f"{hours} Stunden")
    
    if minutes > 0:
        if short:
            parts.append(f"{minutes} Min")
        else:
            parts.append(f"{minutes} Minute" if minutes == 1 else f"{minutes} Minuten")
    
    if secs > 0 or not parts:  # Always show something
        if short:
            parts.append(f"{secs} Sek")
        else:
            parts.append(f"{secs} Sekunde" if secs == 1 else f"{secs} Sekunden")
    
    return " ".join(parts)


def format_duration_simple(seconds: int) -> str:
    """Format seconds to simple German duration (largest unit only).
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Simple duration string using largest applicable unit
        
    Examples:
        format_duration_simple(300) -> "5 Minuten"
        format_duration_simple(7200) -> "2 Stunden"
        format_duration_simple(45) -> "45 Sekunden"
    """
    if seconds >= 3600:
        hours = seconds / 3600
        if hours == int(hours):
            return f"{int(hours)} Stunden" if hours != 1 else "1 Stunde"
        return f"{hours:.1f} Stunden"
    if seconds >= 60:
        minutes = seconds // 60
        return f"{minutes} Minuten" if minutes != 1 else "1 Minute"
    return f"{seconds} Sekunden" if seconds != 1 else "1 Sekunde"
