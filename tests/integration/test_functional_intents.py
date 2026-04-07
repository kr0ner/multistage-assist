"""Functional integration tests for all core intents using natural language.

These tests focus on real-world phrasing and functional groupings.
"""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability

pytestmark = pytest.mark.integration

@pytest.fixture
def keyword_intent_capability(hass, integration_llm_config):
    """Create keyword intent capability instance with real LLM."""
    return KeywordIntentCapability(hass, integration_llm_config)

def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )

class TestLightControl:
    """Natural sentences for lighting control."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_intent,expected_slots", [
        ("Kannst du bitte das Licht in der Küche anmachen?", "HassTurnOn", {"area": "Küche"}),
        ("Ich hätte gerne das Deckenlicht im Flur aus.", "HassTurnOff", {"area": "Flur"}),
        ("Es ist so dunkel im Wohnzimmer, mach mal heller.", "HassLightSet", {"area": "Wohnzimmer", "command": "step_up"}),
        ("Stell das Licht im Bad auf 30 Prozent bitte.", "HassLightSet", {"area": "Bad", "brightness": "30"}),
    ])
    async def test_light_intents(self, keyword_intent_capability, text, expected_intent, expected_slots):
        user_input = make_input(text)
        result = await keyword_intent_capability.run(user_input)
        print(f"\nDEBUG: Input='{text}'")
        print(f"DEBUG: Result={result}")
        
        assert result.get("intent") == expected_intent
        slots = result.get("slots", {})
        for k, v in expected_slots.items():
            actual = str(slots.get(k, "")).strip().lower()
            expected = str(v).strip().lower()
            assert expected in actual or actual in expected, f"Slot '{k}' mismatch. Expected '{expected}', got '{actual}'"

class TestClimateControl:
    """Natural sentences for heating and cooling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_intent,expected_slots", [
        ("Mir ist im Schlafzimmer zu kalt, stell mal die Heizung höher.", "HassClimateSetTemperature", {"area": "Schlafzimmer", "command": "step_up"}),
        ("Heizung in der Küche bitte auf 21 Grad.", "HassClimateSetTemperature", {"area": "Küche", "temperature": "21"}),
        ("Kannst du die Temperatur im Büro auf 22,5 Grad hochdrehen?", "HassClimateSetTemperature", {"area": "Büro", "temperature": "22.5"}),
    ])
    async def test_climate_intents(self, keyword_intent_capability, text, expected_intent, expected_slots):
        user_input = make_input(text)
        result = await keyword_intent_capability.run(user_input)
        print(f"\nDEBUG: Input='{text}'")
        print(f"DEBUG: Result={result}")
        
        assert result.get("intent") == expected_intent
        slots = result.get("slots", {})
        for k, v in expected_slots.items():
            actual = str(slots.get(k, "")).strip().lower()
            expected = str(v).strip().lower()
            assert expected in actual or actual in expected, f"Slot '{k}' mismatch. Expected '{expected}', got '{actual}'"

class TestCoverControl:
    """Natural sentences for covers and blinds."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_intent,expected_slots", [
        ("Fahr das Rollo im Wohnzimmer mal auf die Hälfte runter.", "HassSetPosition", {"area": "Wohnzimmer", "position": "50"}),
        ("Kannst du die Jalousien in der Küche ein bisschen schließen?", "HassSetPosition", {"area": "Küche", "command": "step_down"}),
        ("Mach die Markise auf der Terrasse ganz auf.", "HassTurnOn", {"area": "Terrasse"}),
    ])
    async def test_cover_intents(self, keyword_intent_capability, text, expected_intent, expected_slots):
        user_input = make_input(text)
        result = await keyword_intent_capability.run(user_input)
        print(f"\nDEBUG: Input='{text}'")
        print(f"DEBUG: Result={result}")
        
        assert result.get("intent") == expected_intent
        slots = result.get("slots", {})
        for k, v in expected_slots.items():
            actual = str(slots.get(k, "")).strip().lower()
            expected = str(v).strip().lower()
            assert expected in actual or actual in expected, f"Slot '{k}' mismatch. Expected '{expected}', got '{actual}'"

class TestStateQueries:
    """Natural sentences for asking about the house state."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_intent,expected_slots", [
        ("Brennt das Licht im Keller noch?", "HassGetState", {"area": "Keller"}),
        ("Wie warm ist es eigentlich im Bad?", "HassGetState", {"area": "Bad"}),
        ("Sind alle Fenster im Haus zu?", "HassGetState", {"area": "haus"}),
    ])
    async def test_query_intents(self, keyword_intent_capability, text, expected_intent, expected_slots):
        user_input = make_input(text)
        result = await keyword_intent_capability.run(user_input)
        print(f"\nDEBUG: Input='{text}'")
        print(f"DEBUG: Result={result}")
        
        assert result.get("intent") == expected_intent
        slots = result.get("slots", {})
        for k, v in expected_slots.items():
            if v == "haus": continue # Global keyword
            actual = str(slots.get(k, "")).strip().lower()
            expected = str(v).strip().lower()
            assert expected in actual or actual in expected, f"Slot '{k}' mismatch. Expected '{expected}', got '{actual}'"
