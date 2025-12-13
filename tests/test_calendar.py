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
    
    @pytest.mark.asyncio
    async def test_morgen_resolves_to_actual_date(self, calendar_capability, hass):
        """Test that 'morgen' is resolved to an actual date, not rejected."""
        hass.states.async_entity_ids.return_value = ["calendar.main"]
        main_state = MagicMock()
        main_state.attributes = {"friendly_name": "Main Calendar"}
        hass.states.get = lambda x: main_state if x == "calendar.main" else None
        
        user_input = make_input("Termin morgen")
        
        # Mock LLM returning 'morgen' instead of actual date
        event_data = {
            "summary": "Test",
            "start_date": "morgen",
        }
        
        with patch.object(calendar_capability, "_extract_event_details", return_value=event_data):
            result = await calendar_capability.run(user_input)
        
        # Should proceed to confirmation (date was resolved), not ask again
        assert result.get("status") == "handled"
        assert result.get("pending_data", {}).get("step") == "confirm"
        # The resolved date should be in event_data
        resolved_data = result.get("pending_data", {}).get("event_data", {})
        assert resolved_data.get("start_date") != "morgen"
        # Should be in YYYY-MM-DD format
        from datetime import datetime, timedelta
        expected_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert resolved_data.get("start_date") == expected_date
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_heute(self, calendar_capability):
        """Test that 'heute' resolves to today's date."""
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        
        result = calendar_capability._resolve_relative_dates({"start_date": "heute"})
        assert result["start_date"] == today
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_morgen(self, calendar_capability):
        """Test that 'morgen' resolves to tomorrow's date."""
        from datetime import datetime, timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        result = calendar_capability._resolve_relative_dates({"start_date": "morgen"})
        assert result["start_date"] == tomorrow
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_uebermorgen(self, calendar_capability):
        """Test that 'übermorgen' resolves to day after tomorrow."""
        from datetime import datetime, timedelta
        day_after = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        
        result = calendar_capability._resolve_relative_dates({"start_date": "übermorgen"})
        assert result["start_date"] == day_after
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_in_x_tagen(self, calendar_capability):
        """Test that 'in X Tagen' resolves correctly."""
        from datetime import datetime, timedelta
        target = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")
        
        result = calendar_capability._resolve_relative_dates({"start_date": "in 37 Tagen"})
        assert result["start_date"] == target
        
        # Also test "X Tage" without "in"
        result2 = calendar_capability._resolve_relative_dates({"start_date": "37 Tage"})
        assert result2["start_date"] == target
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_x_tage_with_time(self, calendar_capability):
        """Test that 'X Tage HH:MM' datetime resolves correctly."""
        from datetime import datetime, timedelta
        target_date = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")
        
        result = calendar_capability._resolve_relative_dates({
            "start_date_time": "37 Tage 16:01"
        })
        assert result["start_date_time"] == f"{target_date} 16:01"
    
    @pytest.mark.asyncio
    async def test_resolve_relative_dates_preserves_valid(self, calendar_capability):
        """Test that already valid dates are preserved."""
        result = calendar_capability._resolve_relative_dates({"start_date": "2023-12-25"})
        assert result["start_date"] == "2023-12-25"
    
    @pytest.mark.asyncio
    async def test_validate_dates_with_valid_format(self, calendar_capability):
        """Test _validate_dates with valid date formats."""
        valid_data = {
            "summary": "Test",
            "start_date_time": "2023-12-14 10:00",
            "end_date_time": "2023-12-14 11:00",
        }
        assert calendar_capability._validate_dates(valid_data) is True
        
        valid_all_day = {
            "summary": "Test",
            "start_date": "2023-12-14",
            "end_date": "2023-12-15",
        }
        assert calendar_capability._validate_dates(valid_all_day) is True
    
    @pytest.mark.asyncio
    async def test_validate_dates_with_invalid_format(self, calendar_capability):
        """Test _validate_dates with invalid date formats."""
        invalid_data = {
            "summary": "Test",
            "start_date": "something unparseable",
        }
        assert calendar_capability._validate_dates(invalid_data) is False
        
        invalid_time = {
            "summary": "Test",
            "start_date_time": "tomorrow at 10",
        }
        assert calendar_capability._validate_dates(invalid_time) is False
    
    @pytest.mark.asyncio
    async def test_parse_duration_hours(self, calendar_capability):
        """Test _parse_duration with hours."""
        assert calendar_capability._parse_duration("3 Stunden") == 180
        assert calendar_capability._parse_duration("1 Stunde") == 60
        assert calendar_capability._parse_duration("2 std") == 120
    
    @pytest.mark.asyncio
    async def test_parse_duration_minutes(self, calendar_capability):
        """Test _parse_duration with minutes."""
        assert calendar_capability._parse_duration("30 Minuten") == 30
        assert calendar_capability._parse_duration("45 min") == 45
    
    @pytest.mark.asyncio
    async def test_parse_duration_combined(self, calendar_capability):
        """Test _parse_duration with hours and minutes."""
        assert calendar_capability._parse_duration("2 Stunden 30 Minuten") == 150
        assert calendar_capability._parse_duration("1,5 Stunden") == 90

