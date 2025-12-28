
import pytest
from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability

def test_fuzzy_match_strictness():
    """Test that fuzzy matching enforces strict length equality."""
    # Setup capability without hass dependency
    ki = KeywordIntentCapability(hass=None, config={})
    
    # Test 1: Valid Typo (Swap) - Same Length
    # "lihct" (5) vs "licht" (5)
    # Distance is 2 (two substitutions or swap)
    dist = ki._fuzzy_match_distance("lihct", "licht")
    assert dist is not None, "Should match 'lihct' as typo"
    assert dist <= 2
    
    # Test 2: Invalid Match (Insertion) - Length Mismatch
    # "schalte" (7) vs "schalter" (8)
    # Previous behavior: Matched (dist 1). New behavior: None.
    dist = ki._fuzzy_match_distance("schalte", "schalter")
    assert dist is None, "Should NOT match 'schalte' due to length mismatch"
    
    # Test 3: Invalid Match (Deletion) - Length Mismatch
    # "rolläden" (8) vs "rollläden" (9)
    # User accepted trade-off: this typo is now rejected.
    dist = ki._fuzzy_match_distance("rolläden", "rollläden")
    assert dist is None, "Should fail length check"

    # Test 4: Exact Match
    dist = ki._fuzzy_match_distance("licht", "licht")
    assert dist == 0
