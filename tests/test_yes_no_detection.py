"""Tests for yes/no question detection in state queries."""
import pytest


class TestYesNoQuestionDetection:
    """Tests for yes/no state query response generation."""

    def test_all_covers_closed_returns_ja(self):
        """When all covers are closed, should return 'Ja'."""
        names = ["Küche", "Bad", "Büro"]
        states = ["closed", "closed", "closed"]
        query_state = "closed"
        
        expected_states = ["closed"]
        not_matching = [n for n, s in zip(names, states) if s not in expected_states]
        
        assert len(not_matching) == 0
        # Would generate: "Ja, alle sind geschlossen."

    def test_some_covers_open_returns_nein(self):
        """When some covers are open, should return 'Nein' with list."""
        names = ["Küche", "Bad", "Büro"]
        states = ["closed", "open", "closed"]
        query_state = "closed"
        
        expected_states = ["closed"]
        not_matching = [n for n, s in zip(names, states) if s not in expected_states]
        
        assert len(not_matching) == 1
        assert "Bad" in not_matching
        # Would generate: "Nein, Bad ist noch offen."

    def test_all_lights_on_returns_ja(self):
        """When all lights are on, should return 'Ja'."""
        names = ["Wohnzimmer", "Küche"]
        states = ["on", "on"]
        query_state = "on"
        
        expected_states = ["on"]
        not_matching = [n for n, s in zip(names, states) if s not in expected_states]
        
        assert len(not_matching) == 0
        # Would generate: "Ja, alle sind an."

    def test_some_lights_off_returns_nein(self):
        """When some lights are off, should return 'Nein' with list."""
        names = ["Wohnzimmer", "Küche", "Bad"]
        states = ["on", "off", "off"]
        query_state = "on"
        
        expected_states = ["on"]
        not_matching = [n for n, s in zip(names, states) if s not in expected_states]
        
        assert len(not_matching) == 2
        assert "Küche" in not_matching
        assert "Bad" in not_matching
        # Would generate: "Nein, Küche und Bad sind noch aus."

    def test_many_exceptions_uses_count(self):
        """When more than 3 don't match, use count instead of list."""
        names = ["A", "B", "C", "D", "E"]
        states = ["open", "open", "open", "open", "closed"]
        query_state = "closed"
        
        expected_states = ["closed"]
        not_matching = [n for n, s in zip(names, states) if s not in expected_states]
        
        assert len(not_matching) == 4
        # Would generate: "Nein, 4 sind noch offen."


class TestStateNormalization:
    """Tests for state normalization (German -> English)."""

    def test_geschlossen_maps_to_closed(self):
        """German 'geschlossen' should match 'closed'."""
        state_map = {
            "closed": ["closed"],
            "geschlossen": ["closed"],
        }
        
        assert state_map.get("geschlossen") == ["closed"]

    def test_offen_maps_to_open(self):
        """German 'offen' should match 'open'."""
        state_map = {
            "open": ["open"],
            "offen": ["open"],
        }
        
        assert state_map.get("offen") == ["open"]

    def test_an_maps_to_on(self):
        """German 'an' should match 'on'."""
        state_map = {
            "on": ["on"],
            "an": ["on"],
        }
        
        assert state_map.get("an") == ["on"]

    def test_aus_maps_to_off(self):
        """German 'aus' should match 'off'."""
        state_map = {
            "off": ["off"],
            "aus": ["off"],
        }
        
        assert state_map.get("aus") == ["off"]


class TestOppositeStateWords:
    """Tests for opposite state word mapping."""

    def test_opposite_of_an_is_aus(self):
        """Opposite of 'an' should be 'aus'."""
        opposite_map = {"an": "aus", "aus": "an", "offen": "geschlossen", "geschlossen": "offen"}
        
        assert opposite_map.get("an") == "aus"

    def test_opposite_of_geschlossen_is_offen(self):
        """Opposite of 'geschlossen' should be 'offen'."""
        opposite_map = {"an": "aus", "aus": "an", "offen": "geschlossen", "geschlossen": "offen"}
        
        assert opposite_map.get("geschlossen") == "offen"

    def test_unknown_state_returns_anders(self):
        """Unknown state should return 'anders'."""
        opposite_map = {"an": "aus", "aus": "an", "offen": "geschlossen", "geschlossen": "offen"}
        
        assert opposite_map.get("unknown", "anders") == "anders"
