"""Test the config flow."""

from unittest.mock import patch
import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from multistage_assist.const import DOMAIN
from multistage_assist.config_flow import MultiStageAssistConfigFlow


async def test_user_flow(hass):
    """Test the user flow."""
    # We test the flow class directly below.
    # Testing via hass.config_entries.flow requires complex mocking of DataEntryFlow
    # which is better handled by pytest-homeassistant-custom-component.
    pass

    # Test filling the form
    user_input = {
        "stage1_ip": "192.168.1.100",
        "stage1_port": 1234,
        "stage1_model": "test-model",
        "google_api_key": "secret-key",
        "stage2_model": "gemini-pro",
    }

    # We don't need to patch async_create_entry because MockConfigFlow (base class) implements it
    flow = MultiStageAssistConfigFlow()
    flow.hass = hass
    result = await flow.async_step_user(user_input)

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Multi-Stage Assist"
    assert result["data"] == user_input


# Since we don't have the full HA test suite (pytest-homeassistant-custom-component),
# testing config flows is harder because it relies on the DataEntryFlow manager.
# We will test the class methods directly.


async def test_flow_steps(hass):
    """Test flow steps directly."""
    flow = MultiStageAssistConfigFlow()
    flow.hass = hass

    # Step user - show form
    result = await flow.async_step_user()
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    # Step user - create entry
    user_input = {
        "stage1_ip": "192.168.1.100",
        "stage1_port": 1234,
        "stage1_model": "test-model",
        "google_api_key": "secret-key",
        "stage2_model": "gemini-pro",
    }

    result = await flow.async_step_user(user_input)
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "Multi-Stage Assist"
    assert result["data"] == user_input
