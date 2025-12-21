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
    CONF_STAGE3_MODEL,
    CONF_EMBEDDING_IP,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_MODEL,
    CONF_RERANKER_IP,
    CONF_RERANKER_PORT,
)

# Default embedding model (multilingual, good German support)
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
# Default reranker hostname for HA addon
DEFAULT_RERANKER_HOST = "local-semantic-reranker"


class MultiStageAssistConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """UI step for initial setup."""
        errors = {}

        if user_input is not None:
            # Set embedding defaults to stage1 values if not provided
            if not user_input.get(CONF_EMBEDDING_IP):
                user_input[CONF_EMBEDDING_IP] = user_input.get(CONF_STAGE1_IP, "127.0.0.1")
            if not user_input.get(CONF_EMBEDDING_PORT):
                user_input[CONF_EMBEDDING_PORT] = user_input.get(CONF_STAGE1_PORT, 11434)
            # Set reranker defaults if not provided
            if not user_input.get(CONF_RERANKER_IP):
                user_input[CONF_RERANKER_IP] = DEFAULT_RERANKER_HOST
            if not user_input.get(CONF_RERANKER_PORT):
                user_input[CONF_RERANKER_PORT] = 9876
            return self.async_create_entry(title="Multi-Stage Assist", data=user_input)

        schema = vol.Schema(
            {
                # Stage 1 (Local Control)
                vol.Optional(CONF_STAGE1_IP, default="127.0.0.1"): str,
                vol.Optional(CONF_STAGE1_PORT, default=11434): int,
                vol.Optional(CONF_STAGE1_MODEL, default="qwen3:4b-instruct"): str,
                
                # Stage 3 (Google Gemini Chat)
                vol.Required(CONF_GOOGLE_API_KEY): str,
                vol.Optional(CONF_STAGE3_MODEL, default="gemini-2.5-flash"): str,
                
                # Embedding (Semantic Cache) - defaults to stage1 settings
                vol.Optional(CONF_EMBEDDING_IP, default=""): str,  # Empty = use stage1_ip
                vol.Optional(CONF_EMBEDDING_PORT, default=0): int,  # 0 = use stage1_port
                vol.Optional(CONF_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL): str,
                
                # Reranker (Semantic Cache validation)
                vol.Optional(CONF_RERANKER_IP, default=DEFAULT_RERANKER_HOST): str,
                vol.Optional(CONF_RERANKER_PORT, default=9876): int,
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
            # Set embedding defaults to stage1 values if empty/0
            if not user_input.get(CONF_EMBEDDING_IP):
                user_input[CONF_EMBEDDING_IP] = user_input.get(CONF_STAGE1_IP, "127.0.0.1")
            if not user_input.get(CONF_EMBEDDING_PORT) or user_input.get(CONF_EMBEDDING_PORT) == 0:
                user_input[CONF_EMBEDDING_PORT] = user_input.get(CONF_STAGE1_PORT, 11434)
            # Set reranker defaults if empty/0
            if not user_input.get(CONF_RERANKER_IP):
                user_input[CONF_RERANKER_IP] = DEFAULT_RERANKER_HOST
            if not user_input.get(CONF_RERANKER_PORT) or user_input.get(CONF_RERANKER_PORT) == 0:
                user_input[CONF_RERANKER_PORT] = 9876
            return self.async_create_entry(title="", data=user_input)

        # Use self.config_entry property (provided by base class)
        current_config = {**self.config_entry.data, **self.config_entry.options}
        
        # Get current embedding values, defaulting to stage1 if not set
        current_emb_ip = current_config.get(CONF_EMBEDDING_IP) or current_config.get(CONF_STAGE1_IP, "127.0.0.1")
        current_emb_port = current_config.get(CONF_EMBEDDING_PORT) or current_config.get(CONF_STAGE1_PORT, 11434)
        current_emb_model = current_config.get(CONF_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)
        
        # Get current reranker values
        current_reranker_ip = current_config.get(CONF_RERANKER_IP, DEFAULT_RERANKER_HOST)
        current_reranker_port = current_config.get(CONF_RERANKER_PORT, 9876)

        schema = vol.Schema(
            {
                vol.Optional(CONF_STAGE1_IP, default=current_config.get(CONF_STAGE1_IP, "127.0.0.1")): str,
                vol.Optional(CONF_STAGE1_PORT, default=current_config.get(CONF_STAGE1_PORT, 11434)): int,
                vol.Optional(CONF_STAGE1_MODEL, default=current_config.get(CONF_STAGE1_MODEL, "qwen3:4b-instruct")): str,
                
                vol.Required(CONF_GOOGLE_API_KEY, default=current_config.get(CONF_GOOGLE_API_KEY, "")): str,
                vol.Optional(CONF_STAGE3_MODEL, default=current_config.get(CONF_STAGE3_MODEL, "gemini-2.5-flash")): str,
                
                # Embedding config
                vol.Optional(CONF_EMBEDDING_IP, default=current_emb_ip): str,
                vol.Optional(CONF_EMBEDDING_PORT, default=current_emb_port): int,
                vol.Optional(CONF_EMBEDDING_MODEL, default=current_emb_model): str,
                
                # Reranker config
                vol.Optional(CONF_RERANKER_IP, default=current_reranker_ip): str,
                vol.Optional(CONF_RERANKER_PORT, default=current_reranker_port): int,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)