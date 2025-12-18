"""Integration tests for regression issues with real LLM.

These tests verify fixes for specific bugs that were found in production.
They use real LLM calls to ensure the prompts work correctly.
"""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
from tests.integration import get_llm_config


pytestmark = pytest.mark.integration


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability instance with real LLM."""
    return KeywordIntentCapability(hass, get_llm_config())


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )




class TestAbsoluteBrightness:
    """Tests for brightness commands with specific values.
    
    Bug: "Dimme auf 50%" should have brightness="50", not step_up/step_down.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_brightness", [
        ("Licht auf 50%", "50"),
        ("Dimme das Licht auf 30 Prozent", "30"),
        ("Setze die Helligkeit auf 75%", "75"),
        ("Licht auf 100%", "100"),
    ])
    async def test_absolute_brightness_value(
        self, keyword_intent_capability, text, expected_brightness
    ):
        """Test that absolute brightness values are extracted correctly."""
        user_input = make_input(text)
        
        result = await keyword_intent_capability.run(user_input)
        
        slots = result.get("slots", {})
        brightness = str(slots.get("brightness", ""))
        
        assert brightness == expected_brightness, \
            f"Expected brightness='{expected_brightness}', got '{brightness}'"
        assert result.get("intent") == "HassLightSet"


class TestRelativeBrightness:
    """Tests for relative brightness commands.
    
    Bug: step_up/step_down commands should NOT be cached (tested separately).
    This tests that the LLM produces the correct slot values.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_command", [
        ("Mache das Licht heller", "step_up"),
        ("Licht dunkler", "step_down"),
        ("Heller bitte", "step_up"),
        ("Dunkler machen", "step_down"),
    ])
    async def test_relative_brightness_command(
        self, keyword_intent_capability, text, expected_command
    ):
        """Test that relative brightness uses step_up/step_down."""
        user_input = make_input(text)
        
        result = await keyword_intent_capability.run(user_input)
        
        slots = result.get("slots", {})
        # Check either brightness or command field
        brightness = slots.get("brightness", "")
        command = slots.get("command", "")
        
        has_correct_step = (
            brightness == expected_command or
            command == expected_command
        )
        
        assert has_correct_step, \
            f"Expected '{expected_command}' in slots, got brightness='{brightness}', command='{command}'"


class TestFloorVsArea:
    """Tests for distinguishing floors from areas.
    
    Bug: "Erdgeschoss" should go to floor slot, not area slot.
    """
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_slot,expected_value", [
        ("Licht im Erdgeschoss aus", "floor", "Erdgeschoss"),
        ("Alle Lichter im Obergeschoss an", "floor", "Obergeschoss"),
        ("Licht in der Küche an", "area", "Küche"),
        ("Lichter im Bad aus", "area", "Bad"),
        ("Rolläden im UG hoch", "floor", "UG"),
    ])
    async def test_floor_vs_area_slot(
        self, keyword_intent_capability, text, expected_slot, expected_value
    ):
        """Test that floors go to floor slot, areas to area slot."""
        user_input = make_input(text)
        
        result = await keyword_intent_capability.run(user_input)
        
        slots = result.get("slots", {})
        actual_value = slots.get(expected_slot, "")
        
        # Check that value is in the correct slot (case-insensitive)
        assert expected_value.lower() in actual_value.lower(), \
            f"Expected {expected_slot}='{expected_value}', got slots={slots}"
