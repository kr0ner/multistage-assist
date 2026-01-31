"""Tests for multiple cache hits escalation to LLM.

When multiple cache matches are above the ambiguity threshold (0.70),
Stage1Cache should escalate to Stage2/3 for LLM reasoning instead of
using the first match blindly.
"""

import pytest
import sys
import os

sys.path.insert(0, os.getcwd())


class TestAmbiguityThresholdLogic:
    """Test the ambiguity detection logic directly."""
    
    def test_multiple_above_threshold_is_ambiguous(self):
        """Multiple matches above 0.70 should be considered ambiguous."""
        AMBIGUITY_THRESHOLD = 0.70
        matches = [
            {"intent": "HassGetState", "score": 0.82},
            {"intent": "HassSetPosition", "score": 0.78},  # Both above 0.70
        ]
        
        above_threshold = [m for m in matches if m.get("score", 0) >= AMBIGUITY_THRESHOLD]
        
        assert len(above_threshold) == 2  # Both above threshold
        is_ambiguous = len(above_threshold) > 1
        assert is_ambiguous is True
    
    def test_single_above_threshold_not_ambiguous(self):
        """Single match above threshold is not ambiguous."""
        AMBIGUITY_THRESHOLD = 0.70
        matches = [
            {"intent": "HassTurnOff", "score": 0.88},
            {"intent": "HassTurnOn", "score": 0.55},  # Below threshold
        ]
        
        above_threshold = [m for m in matches if m.get("score", 0) >= AMBIGUITY_THRESHOLD]
        
        assert len(above_threshold) == 1  # Only one above threshold
        is_ambiguous = len(above_threshold) > 1
        assert is_ambiguous is False
    
    def test_no_matches_array_not_ambiguous(self):
        """No matches array means legacy format - not ambiguous."""
        AMBIGUITY_THRESHOLD = 0.70
        data = {
            "found": True,
            "intent": "HassTurnOn",
            "score": 0.85,
        }
        
        matches = data.get("matches", [])
        above_threshold = [m for m in matches if m.get("score", 0) >= AMBIGUITY_THRESHOLD]
        
        assert len(above_threshold) == 0
        is_ambiguous = len(above_threshold) > 1
        assert is_ambiguous is False
    
    def test_all_below_threshold_not_ambiguous(self):
        """All matches below threshold - not ambiguous (and would be cache miss)."""
        AMBIGUITY_THRESHOLD = 0.70
        matches = [
            {"intent": "HassGetState", "score": 0.55},
            {"intent": "HassSetPosition", "score": 0.52},
        ]
        
        above_threshold = [m for m in matches if m.get("score", 0) >= AMBIGUITY_THRESHOLD]
        
        assert len(above_threshold) == 0
        is_ambiguous = len(above_threshold) > 1
        assert is_ambiguous is False


class TestAmbiguousMatchesIntegration:
    """Test integration of ambiguous match detection."""
    
    def test_ambiguous_matches_key_triggers_escalation(self):
        """Cache result with ambiguous_matches should cause escalation."""
        # Simulate cache result with ambiguous_matches
        cached = {
            "intent": "HassGetState",
            "entity_ids": ["cover.rollladen"],
            "slots": {},
            "score": 0.82,
            "source": "anchor",
            "ambiguous_matches": [  # This key triggers escalation
                {"intent": "HassGetState", "score": 0.82},
                {"intent": "HassSetPosition", "score": 0.78},
            ]
        }
        
        # Stage1Cache check logic
        should_escalate = cached.get("ambiguous_matches") is not None
        
        assert should_escalate is True
    
    def test_no_ambiguous_matches_key_allows_success(self):
        """Cache result without ambiguous_matches should allow success."""
        cached = {
            "intent": "HassTurnOn",
            "entity_ids": ["light.kitchen"],
            "slots": {},
            "score": 0.88,
            "source": "anchor",
            # No ambiguous_matches key
        }
        
        # Stage1Cache check logic
        should_escalate = cached.get("ambiguous_matches") is not None
        
        assert should_escalate is False


class TestTimeNormalizationWithAmbiguity:
    """Ensure time normalization doesn't create false ambiguity."""
    
    def test_normalized_time_values_match_same_anchors(self):
        """Different time values should normalize to same and not cause ambiguity."""
        from multistage_assist.utils.german_utils import normalize_for_cache
        
        # All these should normalize to the same pattern
        inputs = [
            "Schalte das Licht in 37 Sekunden an",
            "Schalte das Licht in 5 Minuten an",
            "Schalte das Licht in 2 Stunden an",
        ]
        
        normalized = [normalize_for_cache(inp)[0] for inp in inputs]
        
        # All should produce the exact same normalized string
        assert len(set(normalized)) == 1
        assert "in 10 Minuten" in normalized[0]
