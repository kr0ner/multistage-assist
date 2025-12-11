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
    CONF_STAGE2_MODEL,
)


class MultiStageAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """UI step for initial setup."""
        errors = {}

        if user_input is not None:
            return self.async_create_entry(title="Multi-Stage Assist", data=user_input)

        schema = vol.Schema(
            {
                # Stage 1 (Local Control)
                vol.Optional(CONF_STAGE1_IP, default="127.0.0.1"): str,
                vol.Optional(CONF_STAGE1_PORT, default=5000): int,
                vol.Optional(CONF_STAGE1_MODEL, default="qwen3:4b-instruct"): str,
                
                # Stage 2 (Google Gemini Chat)
                vol.Required(CONF_GOOGLE_API_KEY): str,
                vol.Optional(CONF_STAGE2_MODEL, default="gemini-2.0-flash"): str,
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
        # FIX: Do not set self.config_entry here, it is a read-only property in base class.
        # We can just ignore the passed config_entry as the base class property will 
        # resolve it correctly via self.handler later.
        pass

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Use self.config_entry property (provided by base class)
        current_config = {**self.config_entry.data, **self.config_entry.options}

        schema = vol.Schema(
            {
                vol.Optional(CONF_STAGE1_IP, default=current_config.get(CONF_STAGE1_IP, "127.0.0.1")): str,
                vol.Optional(CONF_STAGE1_PORT, default=current_config.get(CONF_STAGE1_PORT, 5000)): int,
                vol.Optional(CONF_STAGE1_MODEL, default=current_config.get(CONF_STAGE1_MODEL, "qwen3:4b-instruct")): str,
                
                vol.Required(CONF_GOOGLE_API_KEY, default=current_config.get(CONF_GOOGLE_API_KEY, "")): str,
                vol.Optional(CONF_STAGE2_MODEL, default=current_config.get(CONF_STAGE2_MODEL, "gemini-2.0-flash")): str,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)