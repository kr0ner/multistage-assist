"""Tests for CalendarCapability."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation

from multistage_assist.capabilities.calendar import CalendarCapability
from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant instance with calendar entities."""
    hass = MagicMock()
    
    # Mock calendar entities
    hass.states.async_entity_ids.return_value = [
        "calendar.family",
        "calendar.work",
    ]
    
    # Mock calendar states
    family_state = MagicMock()
    family_state.attributes = {"friendly_name": "Family Calendar"}
    
    work_state = MagicMock()
    work_state.attributes = {"friendly_name": "Work Calendar"}
    
    def get_state(entity_id):
        if entity_id == "calendar.family":
            return family_state
        elif entity_id == "calendar.work":
            return work_state
        return None
    
    hass.states.get = get_state
    hass.services.async_call = AsyncMock()
    
    return hass


@pytest.fixture
def calendar_capability(hass):
    """Create calendar capability with real LLM."""
    return CalendarCapability(hass, get_llm_config())


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


class TestCalendarCapability:
    """Tests for CalendarCapability."""
    
    @pytest.mark.asyncio
    async def test_get_calendar_entities(self, calendar_capability, hass):
        """Test that calendar entities are correctly discovered."""
        calendars = calendar_capability._get_calendar_entities()
        
        assert len(calendars) == 2
        assert any(c["entity_id"] == "calendar.family" for c in calendars)
        assert any(c["name"] == "Family Calendar" for c in calendars)
    
    @pytest.mark.asyncio
    async def test_missing_summary_asks(self, calendar_capability):
        """Test that missing summary triggers a question."""
        user_input = make_input("Erstelle einen Termin")
        
        # Mock the LLM to return empty data
        with patch.object(calendar_capability, "_extract_event_details", return_value={}):
            result = await calendar_capability.run(user_input)
        
        assert result.get("status") == "handled"
        assert result.get("pending_data", {}).get("step") == "ask_summary"
        # Check response asks for title
        speech = result["result"].response.speech.get("plain", {}).get("speech", "")
        assert "heißen" in speech.lower() or "termin" in speech.lower()
    
    @pytest.mark.asyncio
    async def test_missing_datetime_asks(self, calendar_capability):
        """Test that missing datetime triggers a question."""
        user_input = make_input("Erstelle einen Termin Zahnarzt")
        
        # Mock the LLM to return only summary
        with patch.object(calendar_capability, "_extract_event_details", return_value={"summary": "Zahnarzt"}):
            result = await calendar_capability.run(user_input)
        
        assert result.get("status") == "handled"
        assert result.get("pending_data", {}).get("step") == "ask_datetime"
    
    @pytest.mark.asyncio
    async def test_multiple_calendars_asks(self, calendar_capability, hass):
        """Test that multiple calendars triggers calendar selection."""
        user_input = make_input("Termin morgen um 10 Uhr")
        
        # Mock complete event data but no calendar selected
        event_data = {
            "summary": "Test",
            "start_date_time": "2023-12-14 10:00",
        }
        
        with patch.object(calendar_capability, "_extract_event_details", return_value=event_data):
            result = await calendar_capability.run(user_input)
        
        assert result.get("status") == "handled"
        assert result.get("pending_data", {}).get("step") == "ask_calendar"
    
    @pytest.mark.asyncio
    async def test_single_calendar_auto_select(self, calendar_capability, hass):
        """Test that single calendar is auto-selected."""
        # Change mock to return only one calendar
        hass.states.async_entity_ids.return_value = ["calendar.main"]
        main_state = MagicMock()
        main_state.attributes = {"friendly_name": "Main Calendar"}
        hass.states.get = lambda x: main_state if x == "calendar.main" else None
        
        user_input = make_input("Termin morgen um 10 Uhr")
        
        event_data = {
            "summary": "Test",
            "start_date_time": "2023-12-14 10:00",
        }
        
        with patch.object(calendar_capability, "_extract_event_details", return_value=event_data):
            result = await calendar_capability.run(user_input)
        
        # Should proceed to confirmation, not ask for calendar
        assert result.get("pending_data", {}).get("step") == "confirm"
    
    @pytest.mark.asyncio
    async def test_confirmation_flow(self, calendar_capability, hass):
        """Test the confirmation flow."""
        # Single calendar setup
        hass.states.async_entity_ids.return_value = ["calendar.main"]
        main_state = MagicMock()
        main_state.attributes = {"friendly_name": "Main Calendar"}
        hass.states.get = lambda x: main_state if x == "calendar.main" else None
        
        # First call - should get to confirmation
        user_input = make_input("Termin morgen um 10 Uhr")
        event_data = {
            "summary": "Zahnarzt",
            "start_date_time": "2023-12-14 10:00",
        }
        
        with patch.object(calendar_capability, "_extract_event_details", return_value=event_data):
            result = await calendar_capability.run(user_input)
        
        assert result.get("pending_data", {}).get("step") == "confirm"
        
        # Confirm with "Ja"
        confirm_input = make_input("Ja")
        pending_data = result.get("pending_data")
        
        result2 = await calendar_capability.continue_flow(confirm_input, pending_data)
        
        assert result2.get("status") == "handled"
        # Should have called the service
        hass.services.async_call.assert_called_once()
        call_args = hass.services.async_call.call_args
        assert call_args[0][0] == "calendar"
        assert call_args[0][1] == "create_event"
    
    @pytest.mark.asyncio
    async def test_cancel_flow(self, calendar_capability, hass):
        """Test canceling the calendar creation."""
        hass.states.async_entity_ids.return_value = ["calendar.main"]
        main_state = MagicMock()
        main_state.attributes = {"friendly_name": "Main Calendar"}
        hass.states.get = lambda x: main_state if x == "calendar.main" else None
        
        cancel_input = make_input("Nein")
        pending_data = {
            "type": "calendar",
            "step": "confirm",
            "event_data": {
                "summary": "Test",
                "start_date_time": "2023-12-14 10:00",
                "calendar_id": "calendar.main",
            },
        }
        
        result = await calendar_capability.continue_flow(cancel_input, pending_data)
        
        assert result.get("status") == "handled"
        speech = result["result"].response.speech.get("plain", {}).get("speech", "")
        assert "nicht erstellt" in speech.lower()
        # Should NOT have called the service
        hass.services.async_call.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_build_confirmation_text(self, calendar_capability):
        """Test confirmation text generation."""
        event_data = {
            "summary": "Zahnarzt",
            "start_date_time": "2023-12-14 10:00",
            "end_date_time": "2023-12-14 11:00",
            "location": "Praxis Dr. Müller",
            "calendar_id": "calendar.family",
        }
        
        text = calendar_capability._build_confirmation_text(event_data)
        
        assert "Zahnarzt" in text
        assert "14.12.2023" in text
        assert "10:00" in text
        assert "Praxis Dr. Müller" in text
