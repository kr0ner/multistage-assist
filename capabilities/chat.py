import logging
from typing import Any, Dict, List

from homeassistant.components import conversation
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class ChatCapability(Capability):
    """
    Handle general conversation/chit-chat when no specific smart home intent is found.
    """

    name = "chat"
    description = "General conversation handler using LLM."

    PROMPT = {
        "system": """
You are a helpful, witty smart home assistant named "Jarvis".
You are chatting with the user in German.
Keep your answers concise (1-2 sentences) and helpful.
Do not hallucinate smart home devices or states you don't know about.
If the user asks something you can't do, politely explain.
""",
        # No strict schema, we want free text.
        # But our PromptExecutor expects a schema usually?
        # If we want free text, we might need to bypass the strict JSON enforcement in PromptExecutor
        # or just ask for a simple JSON wrapper.
        "schema": {
            "properties": {
                "response": {"type": "string"}
            },
            "required": ["response"]
        }
    }

    async def run(self, user_input, **_: Any) -> conversation.ConversationResult:
        text = user_input.text
        
        # TODO: Retrieve history using user_input.conversation_id if available/stored
        # limit to last 500 words as requested
        
        payload = {
            "user_input": text
        }

        _LOGGER.debug("[Chat] Generating chat response for: %s", text)
        
        # We use the existing safe_prompt which enforces JSON.
        # This adds a slight overhead but keeps the architecture consistent.
        data = await self._safe_prompt(self.PROMPT, payload)
        
        response_text = "Ich bin mir nicht sicher, was du meinst."
        if isinstance(data, dict):
            response_text = data.get("response", response_text)

        # Return a conversation result
        intent_response = conversation.intent.IntentResponse(language=user_input.language or "de")
        intent_response.async_set_speech(response_text)
        
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
