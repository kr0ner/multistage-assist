from __future__ import annotations
import logging
import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from homeassistant.components import conversation
from custom_components.multistage_assist.const import (
    CONF_GOOGLE_API_KEY,
    CONF_STAGE2_MODEL,
)

# Import from utils
from custom_components.multistage_assist.conversation_utils import (
    make_response,
    format_chat_history,
)
from .base import Capability

if TYPE_CHECKING:
    from .google_gemini_client import GoogleGeminiClient

_LOGGER = logging.getLogger(__name__)


class ChatCapability(Capability):
    name = "chat"
    description = "General conversation handler using Google Gemini."

    PROMPT = {
        "system": """You are a smart home assistant. Chatting in German.
Decimals use commas. Units written out. Years as words.
""",
        "schema": {
            "properties": {"response": {"type": "string"}},
            "required": ["response"],
        },
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.api_key = config.get(CONF_GOOGLE_API_KEY)
        self.model_name = config.get(CONF_STAGE2_MODEL, "gemini-1.5-flash")
        self._client_wrapper = None
        self._init_lock = asyncio.Lock()

    async def _get_client(self) -> GoogleGeminiClient | None:
        async with self._init_lock:
            if self._client_wrapper:
                return self._client_wrapper
            if not self.api_key:
                return None

            def _create_client():
                from .google_gemini_client import GoogleGeminiClient

                return GoogleGeminiClient(api_key=self.api_key, model=self.model_name)

            try:
                self._client_wrapper = await self.hass.async_add_executor_job(
                    _create_client
                )
            except Exception as e:
                _LOGGER.error("Failed to initialize Google GenAI client: %s", e)
            return self._client_wrapper

    async def run(
        self, user_input, history: List[Dict[str, str]] = None, **_: Any
    ) -> conversation.ConversationResult:
        client = await self._get_client()
        if not client:
            response_text = "Ich bin nicht für Chat konfiguriert."
        else:
            current_text = user_input.text
            # Use utility
            context_str = (
                format_chat_history(history, max_words=500)
                if history
                else f"User: {current_text}"
            )
            _LOGGER.debug("[Chat] Sending %d chars context.", len(context_str))
            response_text = await client.chat(context_str, history)

        # Return conversation result using utility
        return await make_response(response_text, user_input)
