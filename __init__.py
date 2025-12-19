"""Multi-Stage Assist integration."""
from __future__ import annotations

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    CONF_RERANKER_THRESHOLD,
    CONF_HYBRID_ENABLED,
    CONF_HYBRID_ALPHA,
    CONF_HYBRID_NGRAM_SIZE,
    CONF_VECTOR_THRESHOLD,
    CONF_VECTOR_TOP_K,
)

_LOGGER = logging.getLogger(__name__)

# YAML Schema for expert settings (configuration.yaml)
# These are optional - users who want to fine-tune can add them
EXPERT_SCHEMA = vol.Schema({
    vol.Optional(CONF_RERANKER_THRESHOLD): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional(CONF_HYBRID_ENABLED): vol.Boolean(),
    vol.Optional(CONF_HYBRID_ALPHA): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional(CONF_HYBRID_NGRAM_SIZE): vol.All(vol.Coerce(int), vol.Range(min=1, max=5)),
    vol.Optional(CONF_VECTOR_THRESHOLD): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional(CONF_VECTOR_TOP_K): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
})

CONFIG_SCHEMA = vol.Schema(
    {DOMAIN: EXPERT_SCHEMA},
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up from YAML (expert settings only).
    
    The main config is via config flow UI, but expert users can add
    additional tuning options in configuration.yaml:
    
    multistage_assist:
      reranker_threshold: 0.70
      hybrid_enabled: true
      hybrid_alpha: 0.7
      hybrid_ngram_size: 2
      vector_search_threshold: 0.5
      vector_search_top_k: 10
    """
    # Store YAML expert config for later use by config entries
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

    # MERGE CONFIG priority (lowest to highest):
    # 1. YAML expert settings (configuration.yaml)
    # 2. Config entry data (initial setup)
    # 3. Config entry options (reconfiguration via UI)
    yaml_config = hass.data[DOMAIN].get("yaml_config", {})
    effective_config = {**yaml_config, **entry.data, **entry.options}
    
    if yaml_config:
        _LOGGER.debug("[MultiStageAssist] Merged YAML expert config: %s", yaml_config)

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
