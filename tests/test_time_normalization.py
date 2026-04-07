"""Tests for time normalization in cache lookup.

All time expressions (seconds, minutes, hours) should be normalized to "10 Minuten"
for cache lookup, ensuring consistent matching regardless of the actual time value.

normalize_for_cache now PRESERVES articles and prepositions.
Only filler words are stripped.
"""

import pytest
import sys
import os

sys.path.insert(0, os.getcwd())

from multistage_assist.utils.german_utils import normalize_for_cache


class TestTimeNormalization:
    """Test that all time expressions normalize to standard '10 Minuten'."""
    
    def test_normalize_seconds_to_standard_time(self):
        """Seconds should normalize to '10 Minuten'."""
        text = "Schalte das Licht in 37 Sekunden an"
        normalized, extracted = normalize_for_cache(text)
        
        assert "in 10 minuten" in normalized
        assert "sekunden" not in normalized
    
    def test_normalize_minutes_to_standard_time(self):
        """Minutes should normalize to '10 Minuten'."""
        text = "Schalte das Licht in 5 Minuten aus"
        normalized, extracted = normalize_for_cache(text)
        
        assert "in 10 minuten" in normalized
    
    def test_normalize_hours_to_standard_time(self):
        """Hours should normalize to '10 Minuten'."""
        text = "Schalte das Licht in 2 Stunden an"
        normalized, extracted = normalize_for_cache(text)
        
        assert "in 10 minuten" in normalized
        assert "stunden" not in normalized
    
    def test_normalize_duration_seconds_to_standard_time(self):
        """Duration 'für X Sekunden' should normalize to 'für 10 Minuten'."""
        text = "Schalte das Licht für 30 Sekunden an"
        normalized, extracted = normalize_for_cache(text)
        
        assert "fuer 10 minuten" in normalized
    
    def test_normalize_duration_minutes_to_standard_time(self):
        """Duration 'für X Minuten' should normalize to 'für 10 Minuten'."""
        text = "Schalte das Licht für 5 Minuten an"
        normalized, extracted = normalize_for_cache(text)
        
        assert "fuer 10 minuten" in normalized
    
    def test_normalize_duration_hours_to_standard_time(self):
        """Duration 'für X Stunden' should normalize to 'für 10 Minuten'."""
        text = "Schalte das Licht für 2 Stunden an"
        normalized, extracted = normalize_for_cache(text)
        
        assert "fuer 10 minuten" in normalized
    
    def test_normalize_clock_time_to_standard(self):
        """Clock time 'um X Uhr' should normalize to 'um 10 Uhr'."""
        text = "Schalte das Licht um 15:30 Uhr aus"
        normalized, extracted = normalize_for_cache(text)
        
        assert "um 10 uhr" in normalized
    
    def test_normalize_timer_duration_to_standard_time(self):
        """Timer duration 'auf X Minuten' should normalize to '10 Minuten'."""
        text = "Stell einen Timer auf 5 Minuten"
        normalized, extracted = normalize_for_cache(text)
        
        assert "auf 10 minuten" in normalized
    
    def test_extracted_values_preserved(self):
        """Original values should be captured in extracted list."""
        text = "Schalte das Licht in 37 Sekunden an"
        normalized, extracted = normalize_for_cache(text)
        
        # Original value should be in extracted
        assert len(extracted) > 0
    
    def test_non_temporal_text_preserved(self):
        """Text without temporal expressions should be preserved (only canonicalized)."""
        text = "Schalte das Licht im Büro an"
        normalized, extracted = normalize_for_cache(text)
        
        # canonicalize lowercases and replaces umlauts
        assert "schalte das licht im buero an" == normalized
        assert len(extracted) == 0
    
    def test_percentage_normalization(self):
        """Percentage normalization should work."""
        text = "Dimme das Licht auf 30%"
        normalized, extracted = normalize_for_cache(text)
        
        assert "50 prozent" in normalized
        assert 30 in extracted

    def test_articles_preserved_in_time_expressions(self):
        """Articles must be preserved even with time normalization."""
        text = "Schalte das Licht im Badezimmer in 5 Minuten an"
        normalized, extracted = normalize_for_cache(text)
        
        # Articles and prepositions preserved
        assert "das" in normalized
        assert "im" in normalized
        assert "in 10 minuten" in normalized


class TestCacheMatchingConsistency:
    """Test that different time values normalize to same cache key."""
    
    def test_different_seconds_same_normalized(self):
        """Different second values should produce same normalized text."""
        text1 = "Schalte das Licht in 10 Sekunden an"
        text2 = "Schalte das Licht in 37 Sekunden an"
        text3 = "Schalte das Licht in 59 Sekunden an"
        
        norm1, _ = normalize_for_cache(text1)
        norm2, _ = normalize_for_cache(text2)
        norm3, _ = normalize_for_cache(text3)
        
        assert norm1 == norm2 == norm3
    
    def test_different_minutes_same_normalized(self):
        """Different minute values should produce same normalized text."""
        text1 = "Schalte das Licht in 1 Minuten an"
        text2 = "Schalte das Licht in 10 Minuten an"
        text3 = "Schalte das Licht in 45 Minuten an"
        
        norm1, _ = normalize_for_cache(text1)
        norm2, _ = normalize_for_cache(text2)
        norm3, _ = normalize_for_cache(text3)
        
        assert norm1 == norm2 == norm3
    
    def test_seconds_minutes_hours_same_normalized(self):
        """Seconds, minutes, and hours should all normalize to same pattern."""
        text_sec = "Schalte das Licht in 30 Sekunden an"
        text_min = "Schalte das Licht in 5 Minuten an"
        text_hr = "Schalte das Licht in 2 Stunden an"
        
        norm_sec, _ = normalize_for_cache(text_sec)
        norm_min, _ = normalize_for_cache(text_min)
        norm_hr, _ = normalize_for_cache(text_hr)
        
        # All should normalize to same "in 10 Minuten" form
        assert norm_sec == norm_min == norm_hr

    def test_normalize_fractions(self):
        """Test normalization of text fractions to percentages/explicit amounts."""
        # Note: after canonicalize, "ä" becomes "ae", "ü" becomes "ue" etc.
        # Fraction words must match the canonicalized form
        cases = [
            ("Mach das Licht zur Hälfte an", "50 prozent"),
            ("Halb zu", "50 prozent"),
        ]
        for text, expected_text in cases:
            norm, ext = normalize_for_cache(text)
            assert expected_text in norm, f"Expected '{expected_text}' in '{norm}' for input '{text}'"
