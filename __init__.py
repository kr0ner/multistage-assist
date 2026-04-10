"""Multi-Stage Assist integration."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# --- PRE-IMPORT MOCKING ---
# Ensure homeassistant is mocked before any relative imports or sub-imports happen
if "homeassistant" not in sys.modules:
    ha_mock = MagicMock()
    ha_mock.__path__ = [] # Mark as package
    sys.modules["homeassistant"] = ha_mock
    sys.modules["homeassistant.core"] = MagicMock()
    sys.modules["homeassistant.util"] = MagicMock()
    sys.modules["homeassistant.util.dt"] = MagicMock()
    sys.modules["homeassistant.config_entries"] = MagicMock()
    sys.modules["homeassistant.const"] = MagicMock()
    sys.modules["homeassistant.helpers"] = MagicMock()
    sys.modules["homeassistant.helpers.typing"] = MagicMock()

import logging
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    CONF_CACHE_ADDON_IP,
    CONF_CACHE_ADDON_PORT,
    CONF_HYBRID_ENABLED,
    CONF_HYBRID_ALPHA,
    CONF_HYBRID_NGRAM_SIZE,
    CONF_VECTOR_THRESHOLD,
    CONF_VECTOR_TOP_K,
    CONF_CACHE_REGENERATE_ON_STARTUP,
    CONF_CACHE_MAX_ENTRIES,
    CONF_SKIP_STAGE1_LLM,
    CONF_LLM_TIMEOUT,
    CONF_LLM_MAX_RETRIES,
    CONF_DEBUG_CACHE_HITS,
    CONF_DEBUG_LLM_PROMPTS,
)

_LOGGER = logging.getLogger(__name__)

# YAML Schema for expert settings (configuration.yaml)
EXPERT_SCHEMA = vol.Schema({
    vol.Optional(CONF_CACHE_ADDON_IP): str,
    vol.Optional(CONF_CACHE_ADDON_PORT): vol.Coerce(int),
    vol.Optional(CONF_HYBRID_ENABLED): vol.Boolean(),
    vol.Optional(CONF_HYBRID_ALPHA): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional(CONF_HYBRID_NGRAM_SIZE): vol.All(vol.Coerce(int), vol.Range(min=1, max=5)),
    vol.Optional(CONF_VECTOR_THRESHOLD): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional(CONF_VECTOR_TOP_K): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
    vol.Optional(CONF_CACHE_REGENERATE_ON_STARTUP): vol.Boolean(),
    vol.Optional(CONF_CACHE_MAX_ENTRIES): vol.All(vol.Coerce(int), vol.Range(min=100, max=100000)),
    vol.Optional(CONF_SKIP_STAGE1_LLM): vol.Boolean(),
    vol.Optional(CONF_LLM_TIMEOUT): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
    vol.Optional(CONF_LLM_MAX_RETRIES): vol.All(vol.Coerce(int), vol.Range(min=0, max=10)),
    vol.Optional(CONF_DEBUG_CACHE_HITS): vol.Boolean(),
    vol.Optional(CONF_DEBUG_LLM_PROMPTS): vol.Boolean(),
})

CONFIG_SCHEMA = vol.Schema(
    {DOMAIN: EXPERT_SCHEMA},
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up from YAML (expert settings only)."""
    yaml_config = config.get(DOMAIN, {})
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["yaml_config"] = yaml_config
    
    if yaml_config:
        _LOGGER.info("[MultiStageAssist] Loaded expert settings from YAML: %s", yaml_config)
    
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    from .conversation import MultiStageAssistAgent
    from homeassistant.components import conversation

    yaml_config = hass.data[DOMAIN].get("yaml_config", {})
    effective_config = {**yaml_config, **entry.data, **entry.options}
    
    if yaml_config:
        _LOGGER.debug("[MultiStageAssist] Merged YAML expert config: %s", yaml_config)

    agent = MultiStageAssistAgent(hass, effective_config)
    conversation.async_set_agent(hass, entry, agent)
    
    stage1 = agent.stages[1] if len(agent.stages) > 1 else None
    if stage1 and hasattr(stage1, 'has') and stage1.has("semantic_cache"):
        cache = stage1.get("semantic_cache")
        hass.async_create_task(cache.async_startup())
    
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
    await hass.config_entries.async_reload(entry.entry_id)
