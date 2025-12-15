"""Tests for vacuum mode detection in VacuumCapability."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation

from multistage_assist.capabilities.vacuum import VacuumCapability
from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.services.async_call = AsyncMock()
    return hass


@pytest.fixture
def vacuum_capability(hass):
    """Create vacuum capability with real LLM."""
    return VacuumCapability(hass, get_llm_config())


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


class TestVacuumModeDetection:
    """Tests for vacuum mode detection (vacuum vs mop)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_mode", [
        ("Sauge den Keller", "vacuum"),
        ("Staubsauge das Wohnzimmer", "vacuum"),
        ("Saugen im Büro", "vacuum"),
        ("Sauge die Küche", "vacuum"),
    ])
    async def test_vacuum_mode_detection(self, vacuum_capability, input_text, expected_mode):
        """Test that vacuum mode is correctly detected from German text."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        actual_mode = extracted.get("mode", "vacuum")
        assert actual_mode == expected_mode, \
            f"For '{input_text}': expected mode='{expected_mode}', got mode='{actual_mode}'"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text", [
        "Wische die Küche",
        "Wischen im Bad",
        "wische den Keller",
    ])
    async def test_mop_commands(self, vacuum_capability, input_text):
        """Test that mop commands have mode='mop'."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        assert extracted.get("mode") == "mop", \
            f"For '{input_text}': expected mode='mop', got mode='{extracted.get('mode')}'"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_area", [
        ("Sauge den Keller", "Keller"),
        ("Staubsauge das Wohnzimmer", "Wohnzimmer"),
        ("Wische die Küche", "Küche"),
        ("wische den Keller", "Keller"),
    ])
    async def test_vacuum_area_extraction(self, vacuum_capability, input_text, expected_area):
        """Test that area is correctly extracted from vacuum commands."""
        extracted = await vacuum_capability._extract_vacuum_details(input_text)
        
        actual_area = extracted.get("area", "")
        
        # Case-insensitive comparison
        assert actual_area.lower() == expected_area.lower(), \
            f"For '{input_text}': expected area='{expected_area}', got area='{actual_area}'"


class TestVacuumFallback:
    """Test that vacuum defaults work when LLM fails."""
    
    @pytest.mark.asyncio
    async def test_mode_defaults_to_vacuum(self, hass):
        """Test that mode defaults to 'vacuum' if extraction fails."""
        cap = VacuumCapability(hass, {})
        
        # Mock _safe_prompt to return empty dict (LLM failure)
        cap._safe_prompt = AsyncMock(return_value={})
        
        result = await cap._extract_vacuum_details("Sauge irgendwas")
        
        # Should return empty dict, but mode in run() defaults to "vacuum"
        assert result == {}


class TestVacuumRun:
    """Tests for the vacuum run() method."""
    
    @pytest.mark.asyncio
    async def test_vacuum_run_wrong_intent(self, hass):
        """Test that run() returns empty for non-vacuum intents."""
        cap = VacuumCapability(hass, {})
        
        result = await cap.run(
            make_input("test"),
            intent_name="HassTurnOn",
            slots={},
        )
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_vacuum_run_global_scope(self, hass):
        """Test vacuum with global scope (whole house)."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "vacuum",
            "scope": "GLOBAL",
            "area": None,
            "floor": None,
        })
        
        result = await cap.run(
            make_input("Sauge das ganze Haus"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        assert result.get("status") == "handled"
        # Should call the vacuum script
        hass.services.async_call.assert_called_once()
        call_args = hass.services.async_call.call_args
        assert call_args[0][0] == "script"  # domain
        # service_data is the 3rd positional arg
        service_data = call_args[0][2] if len(call_args[0]) > 2 else call_args[1]
        assert service_data["variables"]["target"] == "Alles"
    
    @pytest.mark.asyncio
    async def test_vacuum_run_floor_scope(self, hass):
        """Test vacuum with floor scope."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "vacuum",
            "scope": None,
            "area": None,
            "floor": "Erdgeschoss",
        })
        
        result = await cap.run(
            make_input("Sauge das Erdgeschoss"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        assert result.get("status") == "handled"
        call_args = hass.services.async_call.call_args
        service_data = call_args[0][2] if len(call_args[0]) > 2 else call_args[1]
        assert service_data["variables"]["target"] == "Erdgeschoss"
    
    @pytest.mark.asyncio
    async def test_vacuum_run_room_scope(self, hass):
        """Test vacuum with room scope."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "mop",
            "scope": None,
            "area": "Küche",
            "floor": None,
        })
        cap._normalize_area_name = AsyncMock(return_value="Küche")
        
        result = await cap.run(
            make_input("Wische die Küche"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        assert result.get("status") == "handled"
        call_args = hass.services.async_call.call_args
        service_data = call_args[0][2] if len(call_args[0]) > 2 else call_args[1]
        assert service_data["variables"]["target"] == "Küche"
        assert service_data["variables"]["mode"] == "mop"
    
    @pytest.mark.asyncio
    async def test_vacuum_run_no_target(self, hass):
        """Test vacuum with no target returns error message."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "vacuum",
            "scope": None,
            "area": None,
            "floor": None,
        })
        
        result = await cap.run(
            make_input("Sauge"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        assert result.get("status") == "handled"
        # Should not call the script
        hass.services.async_call.assert_not_called()
        # Should return error message
        response = result.get("result")
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_vacuum_run_script_error(self, hass):
        """Test vacuum handles script errors gracefully."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "vacuum",
            "scope": "GLOBAL",
            "area": None,
            "floor": None,
        })
        
        # Make script call fail
        hass.services.async_call = AsyncMock(side_effect=Exception("Script error"))
        
        result = await cap.run(
            make_input("Sauge das Haus"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        assert result.get("status") == "handled"
        # Should return error message
        response = result.get("result")
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_vacuum_confirmation_message(self, hass):
        """Test vacuum generates proper confirmation message."""
        cap = VacuumCapability(hass, {})
        cap._extract_vacuum_details = AsyncMock(return_value={
            "mode": "vacuum",
            "scope": None,
            "area": "Küche",
            "floor": None,
        })
        cap._normalize_area_name = AsyncMock(return_value="Küche")
        
        result = await cap.run(
            make_input("Sauge die Küche"),
            intent_name="HassVacuumStart",
            slots={},
        )
        
        response = result.get("result")
        if response and hasattr(response, "response") and response.response.speech:
            speech = response.response.speech.get("plain", {}).get("speech", "")
            # Should mention vacuuming/saugen
            assert "saug" in speech.lower() or "Küche" in speech

