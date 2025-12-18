"""Tests for state query response generation."""
import pytest
from utils.response_builder import build_state_response, STATE_DESCRIPTIONS_DE


class TestBuildStateResponse:
    """Tests for the build_state_response function."""

    def test_single_device_on(self):
        """Single device returns 'ist an'."""
        result = build_state_response(["Wohnzimmer"], ["on"], "light")
        assert result == "Wohnzimmer ist an."

    def test_single_device_off(self):
        """Single device returns 'ist aus'."""
        result = build_state_response(["Küche"], ["off"], "light")
        assert result == "Küche ist aus."

    def test_single_cover_closed(self):
        """Single cover returns 'ist geschlossen'."""
        result = build_state_response(["Schlafzimmer"], ["closed"], "cover")
        assert result == "Schlafzimmer ist geschlossen."

    def test_single_cover_open(self):
        """Single cover returns 'ist offen'."""
        result = build_state_response(["Büro"], ["open"], "cover")
        assert result == "Büro ist offen."

    def test_multiple_same_state(self):
        """Multiple devices with same state use 'sind'."""
        result = build_state_response(
            ["Wohnzimmer", "Büro"], ["on", "on"], "light"
        )
        assert result == "Wohnzimmer und Büro sind an."

    def test_multiple_mixed_states(self):
        """Multiple devices with different states grouped correctly."""
        result = build_state_response(
            ["Küche", "Bad", "Büro"], ["on", "off", "on"], "light"
        )
        # Should group by state
        assert "an" in result
        assert "aus" in result
        assert "Küche" in result or "Büro" in result
        assert "Bad" in result

    def test_three_devices_mixed(self):
        """Three devices with mixed states."""
        result = build_state_response(
            ["Küche", "Bad", "Büro"], ["on", "off", "on"], "light"
        )
        # Verify grouped output
        assert "an." in result or "aus." in result

    def test_empty_devices(self):
        """Empty device list returns fallback message."""
        result = build_state_response([], [], "light")
        assert result == "Keine Geräte gefunden."

    def test_unknown_domain_uses_raw_state(self):
        """Unknown domain falls back to raw state value."""
        result = build_state_response(["Device"], ["custom_state"], "unknown_domain")
        assert "custom_state" in result

    def test_climate_states(self):
        """Climate domain uses correct German translations."""
        result = build_state_response(["Heizung"], ["heat"], "climate")
        assert "heizt" in result

    def test_vacuum_states(self):
        """Vacuum domain uses correct German translations."""
        result = build_state_response(["Staubsauger"], ["cleaning"], "vacuum")
        assert "saugt" in result


class TestStateDescriptionsDE:
    """Tests for STATE_DESCRIPTIONS_DE dictionary."""

    def test_light_states_exist(self):
        """Light domain has on/off states."""
        assert "light" in STATE_DESCRIPTIONS_DE
        assert STATE_DESCRIPTIONS_DE["light"]["on"] == "an"
        assert STATE_DESCRIPTIONS_DE["light"]["off"] == "aus"

    def test_cover_states_exist(self):
        """Cover domain has open/closed states."""
        assert "cover" in STATE_DESCRIPTIONS_DE
        assert STATE_DESCRIPTIONS_DE["cover"]["open"] == "offen"
        assert STATE_DESCRIPTIONS_DE["cover"]["closed"] == "geschlossen"

    def test_climate_states_exist(self):
        """Climate domain has heat/cool states."""
        assert "climate" in STATE_DESCRIPTIONS_DE
        assert "heat" in STATE_DESCRIPTIONS_DE["climate"]
        assert "cool" in STATE_DESCRIPTIONS_DE["climate"]
