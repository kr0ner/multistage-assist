from __future__ import annotations  # <--- Essential for type hints without imports

import logging
import asyncio
from typing import Any, Dict, List, TYPE_CHECKING

from homeassistant.components import conversation
from custom_components.multistage_assist.const import CONF_GOOGLE_API_KEY, CONF_STAGE2_MODEL
from .base import Capability

if TYPE_CHECKING:
    # This import is only for type checking/IDEs and won't run at runtime
    from .google_gemini_client import GoogleGeminiClient

_LOGGER = logging.getLogger(__name__)


class ChatCapability(Capability):
    """
    Handle general conversation using Google Gemini (via google-genai SDK).
    Initializes the client lazily to avoid blocking I/O in the event loop.
    """

    name = "chat"
    description = "General conversation handler using Google Gemini."

    PROMPT = {
        "system": """
You are Jarvis, a smart home assistant.
You are chatting with the user in German.
You do not control devices directly in this mode, just chat.
Keep answers concise, helpful, and friendly.
Context from previous messages is provided below.

## Speech Output Rules (CRITICAL for TTS)
1. **Decimals:** Always use a **comma** for decimal numbers (e.g., write "22,5" instead of "22.5").
2. **Units:** Write out units phonetically if needed:
   - "°C" -> "Grad Celsius"
   - "%" -> "Prozent"
   - "kW" -> "Kilowatt"
   - "kWh" -> "Kilowattstunden"
3. **Years:** Write years (1100-1999) as words to ensure correct pronunciation:
   - "1945" -> "neunzehnhundertfünfundvierzig"
   - "2024" -> "zweitausendvierundzwanzig"
4. **General:** Avoid abbreviations. Write "und so weiter" instead of "usw.".
""",
        "schema": {
            "properties": {
                "response": {"type": "string"}
            },
            "required": ["response"]
        }
    }

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.api_key = config.get(CONF_GOOGLE_API_KEY)
        self.model_name = config.get(CONF_STAGE2_MODEL, "gemini-1.5-flash")
        
        self._client_wrapper = None
        self._init_lock = asyncio.Lock()

    async def _get_client(self) -> GoogleGeminiClient | None:
        """
        Initialize the GoogleGeminiClient in an executor to avoid blocking the loop.
        """
        async with self._init_lock:
            if self._client_wrapper:
                return self._client_wrapper
            
            if not self.api_key:
                return None

            # Define the blocking creation function
            def _create_client():
                # --- CRITICAL FIX ---
                # Import inside the thread to prevent blocking the event loop
                from .google_gemini_client import GoogleGeminiClient
                return GoogleGeminiClient(api_key=self.api_key, model=self.model_name)

            try:
                _LOGGER.debug("[Chat] Importing and initializing Gemini Client in background thread...")
                # Run the import and init in a separate thread
                self._client_wrapper = await self.hass.async_add_executor_job(_create_client)
                _LOGGER.debug("[Chat] Gemini Client initialized successfully.")
            except Exception as e:
                _LOGGER.error("Failed to initialize Google GenAI client: %s", e)
                
            return self._client_wrapper

    def _format_history(self, history: List[Dict[str, str]], max_words: int = 500) -> str:
        """Format history into a text block, keeping last ~max_words."""
        full_text = []
        word_count = 0
        
        for turn in reversed(history):
            role = "User" if turn["role"] == "user" else "Jarvis"
            content = turn["content"]
            
            count = len(content.split())
            if word_count + count > max_words:
                break
            
            full_text.insert(0, f"{role}: {content}")
            word_count += count
            
        return "\n".join(full_text)

    async def run(self, user_input, history: List[Dict[str, str]] = None, **_: Any) -> conversation.ConversationResult:
        # Get the client (initializing it if necessary)
        client = await self._get_client()

        if not client:
            _LOGGER.error("[Chat] No Google API key configured.")
            response_text = "Ich bin nicht für Chat konfiguriert."
        else:
            current_text = user_input.text
            context_str = self._format_history(history, max_words=500) if history else f"User: {current_text}"

            _LOGGER.debug("[Chat] Sending context (%d chars) to LLM.", len(context_str))
            
            # The client handles the actual API call
            response_text = await client.chat(context_str, history)

        intent_response = conversation.intent.IntentResponse(language=user_input.language or "de")
        intent_response.async_set_speech(response_text)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
