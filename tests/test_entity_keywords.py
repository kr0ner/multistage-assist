import pytest
from multistage_assist.constants.entity_keywords import (
    TIMER_KEYWORDS,
    CALENDAR_KEYWORDS,
    LIGHT_KEYWORDS,
    _extract_nouns
)

def test_timer_keywords():
    assert "timer" in TIMER_KEYWORDS
    assert "wecker" in TIMER_KEYWORDS

def test_calendar_keywords():
    assert "kalender" in CALENDAR_KEYWORDS
    assert "termin" in CALENDAR_KEYWORDS

def test_extract_nouns():
    keywords = {"das licht": "die lichter"}
    nouns = _extract_nouns(keywords)
    assert "licht" in nouns
    assert "lichter" in nouns

def test_light_keywords_not_empty():
    assert len(LIGHT_KEYWORDS) > 0
