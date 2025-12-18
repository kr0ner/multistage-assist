"""Multi-Stage Assist integration."""
from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up from YAML (not supported)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    from .conversation import MultiStageAssistAgent
    from homeassistant.components import conversation

    # MERGE CONFIG: prefer options (reconfiguration) over data (initial setup)
    # This ensures changes made via "Configure" actually take effect.
    effective_config = {**entry.data, **entry.options}

    agent = MultiStageAssistAgent(hass, effective_config)
    conversation.async_set_agent(hass, entry, agent)
    
    # Initialize semantic cache in background (non-blocking)
    # Stage1 is at index 1 in the stages list
    stage1 = agent.stages[1] if len(agent.stages) > 1 else None
    if stage1 and hasattr(stage1, 'has') and stage1.has("semantic_cache"):
        cache = stage1.get("semantic_cache")
        hass.async_create_task(cache.async_startup())
    
    # REGISTER UPDATE LISTENER: This makes reconfiguration work!
    # When options are updated, this listener triggers a reload of the integration.
    entry.async_on_unload(entry.add_update_listener(update_listener))
    
    _LOGGER.info("Multi-Stage Assist agent registered")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    from homeassistant.components import conversation

    conversation.async_unset_agent(hass, entry)
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True


async def update_listener(hass: HomeAssistant, entry: ConfigEntry):
    """Handle options update."""
    # Reload the integration so the new config is applied immediately
    await hass.config_entries.async_reload(entry.entry_id)
