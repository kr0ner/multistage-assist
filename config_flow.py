"""Config flow for Multi-Stage Assist."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    DOMAIN,
    CONF_STAGE1_IP,
    CONF_STAGE1_PORT,
    CONF_STAGE1_MODEL,
    CONF_GOOGLE_API_KEY,
    CONF_OPENAI_API_KEY,
    CONF_ANTHROPIC_API_KEY,
    CONF_GROK_API_KEY,
    CONF_STAGE3_PROVIDER,
    CONF_STAGE3_MODEL,
    CONF_CACHE_ADDON_IP,
    CONF_CACHE_ADDON_PORT,
    CONF_PROD_CACHE_KEY,
)

# Default cache addon hostname for HA addon
DEFAULT_CACHE_ADDON_HOST = "local-multistage-cache"


class MultiStageAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """UI step for initial setup."""
        errors = {}

        if user_input is not None:

            # Set addon defaults if not provided
            if not user_input.get(CONF_CACHE_ADDON_IP):
                user_input[CONF_CACHE_ADDON_IP] = DEFAULT_CACHE_ADDON_HOST
            if not user_input.get(CONF_CACHE_ADDON_PORT):
                user_input[CONF_CACHE_ADDON_PORT] = 9876
            return self.async_create_entry(title="Multi-Stage Assist", data=user_input)

        schema = vol.Schema(
            {
                # Stage 1 (Local Control)
                vol.Optional(CONF_STAGE1_IP, default="127.0.0.1"): str,
                vol.Optional(CONF_STAGE1_PORT, default=11434): int,
                vol.Optional(CONF_STAGE1_MODEL, default="qwen3:4b-q4_K_M"): str,

                # Stage 3 (Cloud LLM)
                vol.Optional(CONF_STAGE3_PROVIDER, default="gemini"): vol.In(["gemini", "openai", "anthropic", "grok"]),
                vol.Optional(CONF_STAGE3_MODEL, default="gemini-2.0-flash-lite"): str,
                vol.Optional(CONF_GOOGLE_API_KEY, default=""): str,
                vol.Optional(CONF_OPENAI_API_KEY, default=""): str,
                vol.Optional(CONF_ANTHROPIC_API_KEY, default=""): str,
                vol.Optional(CONF_GROK_API_KEY, default=""): str,


                # Semantic Cache Addon
                vol.Optional(CONF_CACHE_ADDON_IP, default=DEFAULT_CACHE_ADDON_HOST): str,
                vol.Optional(CONF_CACHE_ADDON_PORT, default=9876): int,
                vol.Optional(CONF_PROD_CACHE_KEY, default=""): str,
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return MultiStageAssistOptionsFlowHandler(config_entry)


class MultiStageAssistOptionsFlowHandler(config_entries.OptionsFlow):
    """Options flow for editing config (Reconfiguration)."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        pass

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:

            # Set addon defaults if empty/0
            if not user_input.get(CONF_CACHE_ADDON_IP):
                user_input[CONF_CACHE_ADDON_IP] = DEFAULT_CACHE_ADDON_HOST
            if not user_input.get(CONF_CACHE_ADDON_PORT) or user_input.get(CONF_CACHE_ADDON_PORT) == 0:
                user_input[CONF_CACHE_ADDON_PORT] = 9876
            return self.async_create_entry(title="", data=user_input)

        # Use self.config_entry property (provided by base class)
        current_config = {**self.config_entry.data, **self.config_entry.options}


        # Get current addon values
        current_addon_ip = current_config.get(CONF_CACHE_ADDON_IP) or DEFAULT_CACHE_ADDON_HOST
        current_addon_port = current_config.get(CONF_CACHE_ADDON_PORT) or 9876

        schema = vol.Schema(
            {
                vol.Optional(CONF_STAGE1_IP, default=current_config.get(CONF_STAGE1_IP, "127.0.0.1")): str,
                vol.Optional(CONF_STAGE1_PORT, default=current_config.get(CONF_STAGE1_PORT, 11434)): int,
                vol.Optional(CONF_STAGE1_MODEL, default=current_config.get(CONF_STAGE1_MODEL, "qwen3:4b-q4_K_M")): str,

                vol.Optional(CONF_STAGE3_PROVIDER, default=current_config.get(CONF_STAGE3_PROVIDER, "gemini")): vol.In(["gemini", "openai", "anthropic", "grok"]),
                vol.Optional(CONF_STAGE3_MODEL, default=current_config.get(CONF_STAGE3_MODEL, "gemini-2.0-flash-lite")): str,
                vol.Optional(CONF_GOOGLE_API_KEY, default=current_config.get(CONF_GOOGLE_API_KEY, "")): str,
                vol.Optional(CONF_OPENAI_API_KEY, default=current_config.get(CONF_OPENAI_API_KEY, "")): str,
                vol.Optional(CONF_ANTHROPIC_API_KEY, default=current_config.get(CONF_ANTHROPIC_API_KEY, "")): str,
                vol.Optional(CONF_GROK_API_KEY, default=current_config.get(CONF_GROK_API_KEY, "")): str,


                # Cache Addon config
                vol.Optional(CONF_CACHE_ADDON_IP, default=current_addon_ip): str,
                vol.Optional(CONF_CACHE_ADDON_PORT, default=current_addon_port): int,
                vol.Optional(CONF_PROD_CACHE_KEY, default=current_config.get(CONF_PROD_CACHE_KEY, "")): str,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)