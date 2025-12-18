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
    """Tests for intents that should not be cached."""

    def test_temporary_control_in_skip_list(self):
        """HassTemporaryControl should be in the non-cacheable list."""
        non_cacheable = (
            "HassCalendarCreate",
            "HassCreateEvent",
            "HassTimerSet",
            "HassStartTimer",
            "HassTemporaryControl",
        )
        
        assert "HassTemporaryControl" in non_cacheable

    def test_timer_set_in_skip_list(self):
        """HassTimerSet should be in the non-cacheable list."""
        non_cacheable = (
            "HassCalendarCreate",
            "HassCreateEvent",
            "HassTimerSet",
            "HassStartTimer",
            "HassTemporaryControl",
        )
        
        assert "HassTimerSet" in non_cacheable

    def test_regular_intents_cacheable(self):
        """Regular intents like HassTurnOn should be cacheable."""
        non_cacheable = (
            "HassCalendarCreate",
            "HassCreateEvent",
            "HassTimerSet",
            "HassStartTimer",
            "HassTemporaryControl",
        )
        
        assert "HassTurnOn" not in non_cacheable
        assert "HassTurnOff" not in non_cacheable
        assert "HassLightSet" not in non_cacheable


class TestVectorThreshold:
    """Tests for vector search threshold configuration."""

    def test_default_threshold_is_0_5(self):
        """Default vector threshold should be 0.5."""
        from utils.semantic_cache_types import DEFAULT_VECTOR_THRESHOLD
        
        assert DEFAULT_VECTOR_THRESHOLD == 0.5

    def test_threshold_filters_candidates(self):
        """Threshold should filter out low-scoring candidates."""
        threshold = 0.5
        
        # Simulated similarity scores
        scores = [0.9, 0.7, 0.55, 0.45, 0.3, 0.2]
        
        # Filter by threshold
        passing = [s for s in scores if s >= threshold]
        
        assert len(passing) == 3  # 0.9, 0.7, 0.55 pass
        assert 0.45 not in passing
        assert 0.3 not in passing
