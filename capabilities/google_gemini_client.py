"""Google Gemini client wrapper.

Uses lazy initialization to avoid blocking I/O during module import.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

_LOGGER = logging.getLogger(__name__)

# Lazy import - don't import at module load time
_genai = None
_types = None


def _ensure_genai_imported():
    """Lazily import google.genai to avoid blocking I/O at import time."""
    global _genai, _types
    if _genai is None:
        from google import genai
        from google.genai import types
        _genai = genai
        _types = types


class GoogleGeminiClient:
    """Client for Google Gemini API using the new google-genai SDK.
    
    Uses lazy initialization to avoid blocking the event loop during setup.
    """

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        """Store config but defer client creation to avoid blocking I/O."""
        self._api_key = api_key
        self.model = model
        self._client = None
        self._initialized = False

    async def _ensure_client(self):
        """Initialize client in executor to avoid blocking event loop."""
        if self._initialized:
            return
        
        def _init():
            _ensure_genai_imported()
            return _genai.Client(api_key=self._api_key)
        
        loop = asyncio.get_event_loop()
        self._client = await loop.run_in_executor(None, _init)
        self._initialized = True

    @property
    def client(self):
        """Get client (must call _ensure_client first)."""
        return self._client

    def _format_history(self, history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert internal history to google-genai compatible Content list.
        Internal: [{'role': 'user', 'content': '...'}]
        Gemini SDK: [{'role': 'user', 'parts': [{'text': '...'}]}]
        """
        gemini_history = []
        for turn in history:
            # Map 'assistant' role to 'model' for Gemini
            role = "user" if turn["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [{"text": turn["content"]}]
            })
        return gemini_history

    async def chat(self, new_input: str, history: List[Dict[str, str]] = None) -> str:
        """Send a message to Gemini with history using the async client."""
        await self._ensure_client()
        
        if not self._client:
            return "Fehler: Client nicht initialisiert."

        # Prepare full context (History + Current Prompt)
        contents = self._format_history(history or [])
        
        # Add the current user prompt
        contents.append({
            "role": "user",
            "parts": [{"text": new_input}]
        })

        try:
            # Use the async interface (.aio) as per SDK documentation
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=_types.GenerateContentConfig(
                    max_output_tokens=4096,
                    temperature=0.7,
                )
            )
            
            # The response object has a .text property helper
            if response and response.text:
                return response.text
            
            return "Entschuldigung, ich habe keine Antwort erhalten."

        except Exception as e:
            _LOGGER.exception("Gemini API Error: %s", e)
            return f"Entschuldigung, ein Fehler ist aufgetreten: {e}"
