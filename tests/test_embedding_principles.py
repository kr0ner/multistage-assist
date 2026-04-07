import pytest
from multistage_assist.utils.german_utils import normalize_for_cache, canonicalize

def test_principle_1_plurality():
    """Principle 1: Distinguish between plural and singular."""
    norm_singular, _ = normalize_for_cache("Schalte das Licht an")
    norm_plural, _ = normalize_for_cache("Schalte die Lichter an")
    
    # Both article AND noun must remain — articles carry plurality info
    assert "licht" in norm_singular.lower()
    assert "lichter" in norm_plural.lower()
    assert "das" in norm_singular.lower()
    assert "die" in norm_plural.lower()
    assert norm_singular != norm_plural

def test_principle_2_spatial_distinction():
    """Principle 2: Distinguish different areas and floors."""
    norm_keller, _ = normalize_for_cache("Licht im Keller an")
    norm_eg, _ = normalize_for_cache("Licht im Erdgeschoss an")
    
    # Prepositions must be preserved for spatial distinction
    assert "im" in norm_keller.lower()
    assert "keller" in norm_keller.lower()
    assert "im" in norm_eg.lower()
    assert "erdgeschoss" in norm_eg.lower()
    assert norm_keller != norm_eg

def test_principle_2_prepositions_preserved():
    """Principle 2: Different area prepositions must be preserved."""
    norm_balkon, _ = normalize_for_cache("Licht auf dem Balkon an")
    norm_kueche, _ = normalize_for_cache("Licht in der Küche an")
    
    # "auf dem" vs "in der" carry spatial meaning
    assert "auf" in norm_balkon.lower()
    assert "in" in norm_kueche.lower()

def test_principle_3_intent_on_off():
    """Principle 3: 'an' and 'aus' must be distinguishable."""
    norm_on, _ = normalize_for_cache("Schalte das Licht im Badezimmer an")
    norm_off, _ = normalize_for_cache("Schalte das Licht im Badezimmer aus")
    
    assert "an" in norm_on
    assert "aus" in norm_off
    assert norm_on != norm_off

def test_principle_4_articles_preserved():
    """Principle 4: Domain words with articles must be preserved."""
    norm, _ = normalize_for_cache("Schalte das Licht im Badezimmer an")
    
    # Articles are part of the domain representation
    assert "das" in norm.lower()
    assert "licht" in norm.lower()
    assert "im" in norm.lower()

def test_principle_5_opposite_meanings():
    """Principle 5: Opposite meanings MUST be distinguishable."""
    norm_an, _ = normalize_for_cache("Licht an")
    norm_aus, _ = normalize_for_cache("Licht aus")
    
    assert "an" in norm_an.lower()
    assert "aus" in norm_aus.lower()
    assert norm_an != norm_aus

def test_principle_6_multi_command_escalation():
    """Principle 6: Multiple commands map to escalation token."""
    # With "und"
    norm_und, _ = normalize_for_cache("Licht an und Rollo zu")
    assert norm_und == "[MULTIPLE_COMMANDS_ESCALATION]"
    
    # With comma
    norm_comma, _ = normalize_for_cache("Licht an, Rollo zu")
    assert norm_comma == "[MULTIPLE_COMMANDS_ESCALATION]"

def test_principle_7_number_normalization():
    """Principle 7: Numbers are normalized to centroids (50%)."""
    norm_37, ext_37 = normalize_for_cache("Rollo auf 37 Prozent")
    norm_82, ext_82 = normalize_for_cache("Rollo auf 82%")
    
    # Both should normalize to the same string
    assert norm_37 == norm_82
    assert "50 prozent" in norm_37.lower()
    # Original values captured in extracted
    assert 37 in ext_37
    assert 82 in ext_82

def test_filler_words_stripped():
    """Filler words like 'bitte', 'mal' should be stripped."""
    norm, _ = normalize_for_cache("Kannst du bitte das Licht anmachen")
    assert "bitte" not in norm.lower()
    assert "kannst" not in norm.lower()
    # But the meaningful content must remain
    assert "licht" in norm.lower()
    assert "anmachen" in norm.lower()

def test_no_sorting_preserves_order():
    """Verify that we no longer sort tokens (Yoda-speak avoidance)."""
    text = "Schalte das Licht in der Küche an"
    norm, _ = normalize_for_cache(text)
    
    # Order should follow input — articles and prepositions are KEPT now
    expected_contains = ["schalte", "das", "licht", "in", "der", "kueche", "an"]
    for word in expected_contains:
        assert word in norm.lower(), f"'{word}' missing from '{norm}'"
    
    # "an" must be at the end (not sorted alphabetically)
    tokens = norm.lower().split()
    assert tokens[-1] == "an"

def test_area_alias_mapping():
    """Area aliases should be resolved before normalization."""
    # "Bad" → "Badezimmer" (defined in AREA_ALIASES)
    norm, _ = normalize_for_cache("Licht im Bad an")
    assert "badezimmer" in norm.lower()
