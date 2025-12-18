"""Tests for calendar intent detection in keyword_intent."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability with real LLM."""
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


class TestCalendarIntentDetection:
    """Tests for calendar intent detection."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text", [
        "Erstelle einen Kalendereintrag für morgen 15.00 Uhr",
        "Erstelle einen Termin für morgen um 10 Uhr",
        "Termin morgen um 14 Uhr",
        "Kalendereintrag für nächste Woche",
        "Erstelle einen Termin beim Zahnarzt",
        "Termin am Montag um 9 Uhr",
    ])
    async def test_calendar_intent_detected(self, keyword_intent_capability, input_text):
        """Test that calendar commands are detected as calendar domain, not timer."""
        user_input = make_input(input_text)
        
        result = await keyword_intent_capability.run(user_input)
        
        assert result.get("domain") == "calendar", \
            f"For '{input_text}': expected domain='calendar', got domain='{result.get('domain')}'"
        assert result.get("intent") in ("HassCalendarCreate", "HassCreateEvent"), \
            f"For '{input_text}': expected calendar intent, got intent='{result.get('intent')}'"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text", [
        "Timer für 5 Minuten",
        "Stelle einen Timer auf 10 Minuten",
        "Wecker in 30 Minuten",
        "Starte einen Countdown für 2 Minuten",
    ])
    async def test_timer_intent_still_works(self, keyword_intent_capability, input_text):
        """Test that timer commands are still detected correctly."""
        user_input = make_input(input_text)
        
        result = await keyword_intent_capability.run(user_input)
        
        assert result.get("domain") == "timer", \
            f"For '{input_text}': expected domain='timer', got domain='{result.get('domain')}'"
        assert result.get("intent") == "HassTimerSet", \
            f"For '{input_text}': expected intent='HassTimerSet', got intent='{result.get('intent')}'"


class TestDomainPriority:
    """Test that domain detection priority is correct."""
    
    def test_calendar_priority_over_timer(self, keyword_intent_capability):
        """Calendar keywords should have priority over generic time words."""
        # This tests the _detect_domain method
        cap = keyword_intent_capability
        
        # "Uhr" alone should NOT trigger timer anymore
        result = cap._detect_domain("Meeting um 15 Uhr")
        assert result is None or result != "timer", \
            "'Uhr' alone should not trigger timer domain"
        
        # Calendar keywords should definitely trigger calendar
        result = cap._detect_domain("Termin morgen")
        assert result == "calendar"
        
        result = cap._detect_domain("Kalendereintrag erstellen")
        assert result == "calendar"
        
        # Timer keywords should still work
        result = cap._detect_domain("Timer für 5 Minuten")
        assert result == "timer"
        
        result = cap._detect_domain("Wecker in 10 Minuten")
        assert result == "timer"
