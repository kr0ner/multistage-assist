import pytest
from multistage_assist.constants.entity_keywords import (
    TIMER_KEYWORDS,
    CALENDAR_KEYWORDS,
    LIGHT_KEYWORDS,
    COVER_KEYWORDS,
    GENERIC_NAMES,
    ON_INDICATORS,
    OFF_INDICATORS,
    DOMAIN_NAMES,
    _extract_nouns,
)

def test_extract_nouns_from_keyword_dict():
    """Test noun extraction strips articles correctly."""
    keywords = {"das licht": "die lichter", "der schalter": "die schalter"}
    nouns = _extract_nouns(keywords)
    assert "licht" in nouns
    assert "lichter" in nouns
    assert "schalter" in nouns
    assert "das" not in nouns
    assert "die" not in nouns

def test_generic_names_include_all_domains():
    """Test that GENERIC_NAMES covers nouns from all domain keyword dicts."""
    assert "licht" in GENERIC_NAMES
    assert "rollladen" in GENERIC_NAMES
    assert "schalter" in GENERIC_NAMES or "steckdose" in GENERIC_NAMES

def test_on_off_indicators_are_disjoint():
    """Test that ON and OFF indicators don't overlap."""
    assert ON_INDICATORS.isdisjoint(OFF_INDICATORS)

def test_domain_names_coverage():
    """Test that domain names cover essential domains."""
    assert "light" in DOMAIN_NAMES
    assert "cover" in DOMAIN_NAMES
    assert "climate" in DOMAIN_NAMES
    assert "switch" in DOMAIN_NAMES
