"""Test the Multi-Stage Assist initialization."""

from unittest.mock import patch, MagicMock
import pytest
from homeassistant.const import CONF_PLATFORM
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

# Import the integration
# We need to import it as a module to support relative imports within it
from multistage_assist import async_setup_entry, async_unload_entry, DOMAIN


async def test_setup_entry(hass, config_entry, mock_conversation):
    """Test setting up the integration from a config entry."""

    # Mock MultiStageAssistAgent to avoid instantiating real stages for this test
    with patch(
        "multistage_assist.conversation.MultiStageAssistAgent"
    ) as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent

        assert await async_setup_entry(hass, config_entry)

        # Verify agent was initialized
        mock_agent_cls.assert_called_once()

        # Verify agent was registered with conversation component
        mock_conversation["async_set_agent"].assert_called_once_with(
            hass, config_entry, mock_agent
        )

        # Verify update listener was added
        config_entry.add_update_listener.assert_called_once()

        # Verify data stored in hass
        assert DOMAIN in hass.data
        assert hass.data[DOMAIN][config_entry.entry_id] == config_entry.data


async def test_unload_entry(hass, config_entry, mock_conversation):
    """Test unloading the integration."""
    # Setup first
    hass.data[DOMAIN] = {config_entry.entry_id: config_entry.data}

    assert await async_unload_entry(hass, config_entry)

    # Verify agent was unregistered
    mock_conversation["async_unset_agent"].assert_called_once_with(hass, config_entry)

    # Verify data removed
    assert config_entry.entry_id not in hass.data[DOMAIN]
