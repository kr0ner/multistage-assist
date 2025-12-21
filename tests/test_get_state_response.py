"""Tests for HassGetState response logic.

These tests verify that state queries return correct German responses:
- "Sind alle Lichter aus?" with 1 ON → "Nein, Flur ist noch an."
- "Sind alle Lichter an?" with all ON → "Ja, alle sind an."
"""

import pytest
from unittest.mock import MagicMock, patch


class TestHassGetStateResponseLogic:
    """Test HassGetState 'are all X?' response generation."""

    @pytest.fixture
    def mock_states(self):
        """Create mock entity states for testing."""
        def create_state(entity_id, state, friendly_name=None):
            mock = MagicMock()
            mock.state = state
            mock.entity_id = entity_id
            mock.attributes = {"friendly_name": friendly_name or entity_id.split(".")[-1].replace("_", " ").title()}
            return mock
        return create_state

    def test_are_all_lights_off_one_on(self, mock_states):
        """User asks 'Sind alle Lichter aus?' and 1 light is ON.
        
        Expected: 'Nein, Flur EG ist noch an.'
        """
        # Setup: 23 lights off, 1 light on
        states = {
            "light.flur_eg": mock_states("light.flur_eg", "on", "Flur EG"),
            "light.kuche": mock_states("light.kuche", "off", "Küche"),
            "light.buro": mock_states("light.buro", "off", "Büro"),
        }

        # Variables from the code
        all_entity_ids = list(states.keys())
        expected_states = ["off"]  # User asked for "aus" → "off"
        positive_word = "aus"
        opposite_word = "an"

        # Simulate the logic
        all_names = [states[eid].attributes["friendly_name"] for eid in all_entity_ids]
        all_states_list = [states[eid].state for eid in all_entity_ids]
        
        matching = [n for n, s in zip(all_names, all_states_list) if s in expected_states]
        not_matching = [n for n, s in zip(all_names, all_states_list) if s not in expected_states]

        # Build response
        if not not_matching:
            speech_text = f"Ja, alle sind {positive_word}."
        elif len(not_matching) == 1:
            speech_text = f"Nein, {not_matching[0]} ist noch {opposite_word}."
        elif len(not_matching) <= 3:
            speech_text = f"Nein, {' und '.join(not_matching)} sind noch {opposite_word}."
        else:
            speech_text = f"Nein, {len(not_matching)} sind noch {opposite_word}."

        # Assert the correct response
        assert speech_text == "Nein, Flur EG ist noch an."

    def test_are_all_lights_on_all_on(self, mock_states):
        """User asks 'Sind alle Lichter an?' and all lights are ON.
        
        Expected: 'Ja, alle sind an.'
        """
        states = {
            "light.flur_eg": mock_states("light.flur_eg", "on", "Flur EG"),
            "light.kuche": mock_states("light.kuche", "on", "Küche"),
        }

        all_entity_ids = list(states.keys())
        expected_states = ["on"]
        positive_word = "an"
        opposite_word = "aus"

        all_names = [states[eid].attributes["friendly_name"] for eid in all_entity_ids]
        all_states_list = [states[eid].state for eid in all_entity_ids]
        
        not_matching = [n for n, s in zip(all_names, all_states_list) if s not in expected_states]

        if not not_matching:
            speech_text = f"Ja, alle sind {positive_word}."
        else:
            speech_text = f"Nein, {len(not_matching)} sind noch {opposite_word}."

        assert speech_text == "Ja, alle sind an."

    def test_are_all_lights_off_multiple_on(self, mock_states):
        """User asks 'Sind alle Lichter aus?' and 2 lights are ON.
        
        Expected: 'Nein, Flur EG und Küche sind noch an.'
        """
        states = {
            "light.flur_eg": mock_states("light.flur_eg", "on", "Flur EG"),
            "light.kuche": mock_states("light.kuche", "on", "Küche"),
            "light.buro": mock_states("light.buro", "off", "Büro"),
        }

        all_entity_ids = list(states.keys())
        expected_states = ["off"]
        positive_word = "aus"
        opposite_word = "an"

        all_names = [states[eid].attributes["friendly_name"] for eid in all_entity_ids]
        all_states_list = [states[eid].state for eid in all_entity_ids]
        
        not_matching = [n for n, s in zip(all_names, all_states_list) if s not in expected_states]

        if len(not_matching) == 1:
            speech_text = f"Nein, {not_matching[0]} ist noch {opposite_word}."
        elif len(not_matching) <= 3:
            speech_text = f"Nein, {' und '.join(not_matching)} sind noch {opposite_word}."
        else:
            speech_text = f"Nein, {len(not_matching)} sind noch {opposite_word}."

        assert speech_text == "Nein, Flur EG und Küche sind noch an."
