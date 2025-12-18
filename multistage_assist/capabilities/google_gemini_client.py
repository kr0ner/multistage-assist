import logging
from typing import List, Dict, Any

# Import the new SDK
from google import genai
from google.genai import types

_LOGGER = logging.getLogger(__name__)

class GoogleGeminiClient:
    """Client for Google Gemini API using the new google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        """
        Initialize the client. 
        The SDK handles API keys, but we pass it explicitly to support HA config.
        """
        self.model = model
        # Initialize the client. API key can be passed directly.
        self.client = genai.Client(api_key=api_key)

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
        """
        Send a message to Gemini with history using the async client.
        """
        if not self.client:
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
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
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
