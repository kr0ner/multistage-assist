"""Tests for the StepControlCapability.

Tests cover:
- Light brightness step up/down
- Cover position step up/down
- Fan percentage step up/down
- Climate temperature step up/down
- Off-state handling
- Edge cases (min/max values)
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional


# Mock HomeAssistant state object
@dataclass
class MockState:
    entity_id: str
    state: str
    attributes: Dict[str, Any]


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    return hass


@pytest.fixture
def step_control(mock_hass):
    """Create StepControlCapability instance."""
    from multistage_assist.capabilities.step_control import StepControlCapability
    return StepControlCapability(mock_hass, {})


class TestLightBrightnessStep:
    """Test light brightness step operations."""

    @pytest.mark.asyncio
    async def test_step_up_from_50_percent(self, step_control, mock_hass):
        """Step up from 50% should increase by calculated step."""
        # 50% brightness = 127.5 in 0-255 range
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="on",
            attributes={"brightness": 128}  # ~50%
        )
        
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="step_up",
            domain="light"
        )
        
        assert result["new_value"] > 50
        assert result["current_value"] == 50
        assert result["attribute"] == "brightness"
        # Step should be 35% of 50 = 17.5, min 10, so +17
        assert result["step_applied"] >= 10

    @pytest.mark.asyncio
    async def test_step_up_from_off(self, step_control, mock_hass):
        """Step up from off should turn on to default brightness."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="off",
            attributes={"brightness": 0}
        )
        
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="step_up",
            domain="light"
        )
        
        # Should turn on to off_to_on value (50 from domain_config.py)
        assert result["new_value"] == 50
        assert result["current_value"] == 0


    @pytest.mark.asyncio
    async def test_step_down_to_zero(self, step_control, mock_hass):
        """Step down from low brightness should go to 0."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="on",
            attributes={"brightness": 25}  # ~10%
        )
        
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="step_down",
            domain="light"
        )
        
        assert result["new_value"] == 0
        assert result["current_value"] < 15

    @pytest.mark.asyncio
    async def test_step_up_caps_at_100(self, step_control, mock_hass):
        """Step up from high brightness should cap at 100%."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="on",
            attributes={"brightness": 250}  # ~98%
        )
        
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="step_up",
            domain="light"
        )
        
        assert result["new_value"] == 100


class TestCoverPositionStep:
    """Test cover position step operations."""

    @pytest.mark.asyncio
    async def test_cover_step_up(self, step_control, mock_hass):
        """Step up on cover should increase position."""
        mock_hass.states.get.return_value = MockState(
            entity_id="cover.test",
            state="open",
            attributes={"current_position": 50}
        )
        
        result = await step_control.run(
            None,
            entity_id="cover.test",
            command="step_up",
            domain="cover"
        )
        
        assert result["new_value"] > 50
        assert result["domain"] == "cover"

    @pytest.mark.asyncio
    async def test_cover_step_up_from_closed(self, step_control, mock_hass):
        """Step up from closed cover should open to default."""
        mock_hass.states.get.return_value = MockState(
            entity_id="cover.blinds",
            state="closed",
            attributes={"current_position": 0}
        )
        
        result = await step_control.run(
            None,
            entity_id="cover.blinds",
            command="step_up",
            domain="cover"
        )
        
        # Cover off_to_on is 100 (fully open)
        assert result["new_value"] == 100

    @pytest.mark.asyncio
    async def test_cover_step_down(self, step_control, mock_hass):
        """Step down on cover should decrease position."""
        mock_hass.states.get.return_value = MockState(
            entity_id="cover.test",
            state="open",
            attributes={"current_position": 80}
        )
        
        result = await step_control.run(
            None,
            entity_id="cover.test",
            command="step_down",
            domain="cover"
        )
        
        assert result["new_value"] < 80
        assert result["new_value"] >= 0


class TestFanPercentageStep:
    """Test fan speed step operations."""

    @pytest.mark.asyncio
    async def test_fan_step_up(self, step_control, mock_hass):
        """Step up on fan should increase percentage."""
        mock_hass.states.get.return_value = MockState(
            entity_id="fan.test",
            state="on",
            attributes={"percentage": 50}
        )
        
        result = await step_control.run(
            None,
            entity_id="fan.test",
            command="step_up",
            domain="fan"
        )
        
        assert result["new_value"] > 50
        assert result["domain"] == "fan"

    @pytest.mark.asyncio
    async def test_fan_step_up_from_off(self, step_control, mock_hass):
        """Step up from off fan should turn on to default."""
        mock_hass.states.get.return_value = MockState(
            entity_id="fan.living_room",
            state="off",
            attributes={"percentage": 0}
        )
        
        result = await step_control.run(
            None,
            entity_id="fan.living_room",
            command="step_up",
            domain="fan"
        )
        
        # Fan off_to_on is 50
        assert result["new_value"] == 50


class TestClimateTemperatureStep:
    """Test climate temperature step operations."""

    @pytest.mark.asyncio
    async def test_climate_step_up(self, step_control, mock_hass):
        """Step up on climate should increase temperature."""
        mock_hass.states.get.return_value = MockState(
            entity_id="climate.thermostat",
            state="heat",
            attributes={"temperature": 20.0, "current_temperature": 19.5}
        )
        
        result = await step_control.run(
            None,
            entity_id="climate.thermostat",
            command="step_up",
            domain="climate"
        )
        
        # Default step is 1.0
        assert result["new_value"] == 21.0
        assert result["current_value"] == 20.0
        assert result["step_applied"] == 1.0

    @pytest.mark.asyncio
    async def test_climate_step_down(self, step_control, mock_hass):
        """Step down on climate should decrease temperature."""
        mock_hass.states.get.return_value = MockState(
            entity_id="climate.thermostat",
            state="heat",
            attributes={"temperature": 22.0}
        )
        
        result = await step_control.run(
            None,
            entity_id="climate.thermostat",
            command="step_down",
            domain="climate"
        )
        
        assert result["new_value"] == 21.0

    @pytest.mark.asyncio
    async def test_climate_step_up_caps_at_max(self, step_control, mock_hass):
        """Step up at max temperature should stay at max."""
        mock_hass.states.get.return_value = MockState(
            entity_id="climate.thermostat",
            state="heat",
            attributes={"temperature": 28.0}  # max_temp default
        )
        
        result = await step_control.run(
            None,
            entity_id="climate.thermostat",
            command="step_up",
            domain="climate"
        )
        
        assert result["new_value"] == 28.0

    @pytest.mark.asyncio
    async def test_climate_step_down_caps_at_min(self, step_control, mock_hass):
        """Step down at min temperature should stay at min."""
        mock_hass.states.get.return_value = MockState(
            entity_id="climate.thermostat",
            state="heat",
            attributes={"temperature": 16.0}  # min_temp default
        )
        
        result = await step_control.run(
            None,
            entity_id="climate.thermostat",
            command="step_down",
            domain="climate"
        )
        
        assert result["new_value"] == 16.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_command(self, step_control, mock_hass):
        """Invalid command should return empty dict."""
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="invalid",
            domain="light"
        )
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_entity_not_found(self, step_control, mock_hass):
        """Missing entity should return empty dict."""
        mock_hass.states.get.return_value = None
        
        result = await step_control.run(
            None,
            entity_id="light.nonexistent",
            command="step_up",
            domain="light"
        )
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_unsupported_domain(self, step_control, mock_hass):
        """Unsupported domain should return empty dict."""
        mock_hass.states.get.return_value = MockState(
            entity_id="sensor.test",
            state="25",
            attributes={}
        )
        
        result = await step_control.run(
            None,
            entity_id="sensor.test",
            command="step_up",
            domain="sensor"
        )
        
        # Sensor has no step config
        assert result == {}

    @pytest.mark.asyncio
    async def test_auto_detect_domain(self, step_control, mock_hass):
        """Domain should be auto-detected from entity_id."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.kitchen",
            state="on",
            attributes={"brightness": 128}
        )
        
        result = await step_control.run(
            None,
            entity_id="light.kitchen",
            command="step_up",
            # No domain provided
        )
        
        assert result["domain"] == "light"
        assert result["new_value"] > 50

    @pytest.mark.asyncio
    async def test_step_down_on_off_entity(self, step_control, mock_hass):
        """Step down on off entity should return empty."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="off",
            attributes={"brightness": 0}
        )
        
        result = await step_control.run(
            None,
            entity_id="light.test",
            command="step_down",
            domain="light"
        )
        
        # Nothing to step down from
        assert result == {}


class TestApplyStepConvenience:
    """Test the apply_step convenience method."""

    @pytest.mark.asyncio
    async def test_apply_step_returns_tuple(self, step_control, mock_hass):
        """apply_step should return (attribute, value) tuple."""
        mock_hass.states.get.return_value = MockState(
            entity_id="light.test",
            state="on",
            attributes={"brightness": 128}
        )
        
        attr, value = await step_control.apply_step(
            entity_id="light.test",
            command="step_up"
        )
        
        assert attr == "brightness"
        assert value > 50

    @pytest.mark.asyncio
    async def test_apply_step_returns_none_on_failure(self, step_control, mock_hass):
        """apply_step should return (None, None) on failure."""
        mock_hass.states.get.return_value = None
        
        attr, value = await step_control.apply_step(
            entity_id="light.nonexistent",
            command="step_up"
        )
        
        assert attr is None
        assert value is None
