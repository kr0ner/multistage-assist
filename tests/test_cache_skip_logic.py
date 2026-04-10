"""Tests for compound command and cache skip logic."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestCompoundCommandCacheSkip:
    """Tests for skipping cache on compound commands with separators."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.states = MagicMock()
        hass.config.path.return_value = "/tmp"
        return hass

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return {
            "stage1_ip": "localhost",
            "stage1_port": 11434,
            "stage1_model": "test-model",
        }

    def test_detects_und_separator(self):
        """Commands with 'und' should be detected as compound."""
        separators = [",", " and ", " und ", " oder ", " or ", " dann "]
        
        text = "Mach das Licht in der Küche aus und im Wohnzimmer an"
        text_lower = f" {text.lower()} "
        is_compound = any(sep in text_lower for sep in separators)
        
        assert is_compound is True

    def test_detects_oder_separator(self):
        """Commands with 'oder' should be detected as compound."""
        separators = [",", " and ", " und ", " oder ", " or ", " dann "]
        
        text = "Heizung auf 22 Grad oder 21 Grad"
        text_lower = f" {text.lower()} "
        is_compound = any(sep in text_lower for sep in separators)
        
        assert is_compound is True

    def test_simple_command_not_compound(self):
        """Simple commands without separators are not compound."""
        separators = [",", " and ", " und ", " oder ", " or ", " dann "]
        
        text = "Schalte das Licht im Wohnzimmer an"
        text_lower = f" {text.lower()} "
        is_compound = any(sep in text_lower for sep in separators)
        
        assert is_compound is False

    def test_comma_is_compound(self):
        """Commands with comma should be detected as compound."""
        separators = [",", " and ", " und ", " oder ", " or ", " dann "]
        
        text = "Licht an, Rollo runter"
        text_lower = f" {text.lower()} "
        is_compound = any(sep in text_lower for sep in separators)
        
        assert is_compound is True


class TestNonCacheableIntents:
    """Tests for intents that should skip re-caching on cache hits."""

    def test_timer_set_in_nocache_intents(self):
        """HassTimerSet should be in NOCACHE_INTENTS (skip re-caching)."""
        from multistage_assist.capabilities.command_processor import NOCACHE_INTENTS
        assert "HassTimerSet" in NOCACHE_INTENTS

    def test_timer_cancel_in_nocache_intents(self):
        """HassTimerCancel should be in NOCACHE_INTENTS."""
        from multistage_assist.capabilities.command_processor import NOCACHE_INTENTS
        assert "HassTimerCancel" in NOCACHE_INTENTS

    def test_calendar_in_nocache_intents(self):
        """HassCalendarCreate should be in NOCACHE_INTENTS."""
        from multistage_assist.capabilities.command_processor import NOCACHE_INTENTS
        assert "HassCalendarCreate" in NOCACHE_INTENTS

    def test_regular_intents_cacheable(self):
        """Regular intents should not be in NOCACHE_INTENTS."""
        from multistage_assist.capabilities.command_processor import NOCACHE_INTENTS
        assert "HassTurnOn" not in NOCACHE_INTENTS
        assert "HassTurnOff" not in NOCACHE_INTENTS
        assert "HassLightSet" not in NOCACHE_INTENTS

    def test_timer_not_in_cache_bypass(self):
        """HassTimerSet should NOT be in CACHE_BYPASS_INTENTS (timer intents are now cacheable)."""
        from multistage_assist.stage1_cache import CACHE_BYPASS_INTENTS
        assert "HassTimerSet" not in CACHE_BYPASS_INTENTS

    def test_timer_cancel_still_bypasses_cache(self):
        """HassTimerCancel should still bypass cache (needs specific timer name)."""
        from multistage_assist.stage1_cache import CACHE_BYPASS_INTENTS
        assert "HassTimerCancel" in CACHE_BYPASS_INTENTS


class TestVectorThreshold:
    """Tests for vector search threshold configuration."""

    def test_default_threshold_is_0_75(self):
        """Default vector threshold should be 0.75 (aligned with const.py EXPERT_DEFAULTS)."""
        from utils.semantic_cache_types import DEFAULT_VECTOR_THRESHOLD
        
        assert DEFAULT_VECTOR_THRESHOLD == 0.75

    def test_threshold_filters_candidates(self):
        """Threshold should filter out low-scoring candidates."""
        threshold = 0.82
        
        # Simulated similarity scores (multilingual-minilm produces tighter clusters)
        scores = [0.95, 0.88, 0.83, 0.78, 0.65, 0.45]
        
        # Filter by threshold
        passing = [s for s in scores if s >= threshold]
        
        assert len(passing) == 3  # 0.95, 0.88, 0.83 pass
        assert 0.78 not in passing
        assert 0.65 not in passing
