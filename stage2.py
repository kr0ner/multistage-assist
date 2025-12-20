import logging
from typing import Dict, List, Any
from .base_stage import BaseStage
from .capabilities.chat import ChatCapability

_LOGGER = logging.getLogger(__name__)

class Stage2Processor(BaseStage):
    """
    Stage 2: General Conversation (Chat).
    Acts as the final fallback and manages sticky chat sessions via Google Gemini.
    """
    name = "stage2"
    capabilities = [ChatCapability]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        # Session storage: conversation_id -> List[history]
        self._chat_sessions: Dict[str, List[Dict[str, str]]] = {}

    def has_active_chat(self, user_input) -> bool:
        """Check if this conversation ID is already in chat mode."""
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        return key in self._chat_sessions

    async def process(self, user_input, context=None):
        """New unified interface - wraps legacy run() for backward compatibility."""
        from .stage_result import StageResult
        result = await self.run(user_input, prev_result=context)
        if result.get("status") == "handled":
            return StageResult(
                status="success",
                response=result.get("result"),
                raw_text=user_input.text,
            )
        return StageResult.error(
            response=result.get("result"),
            raw_text=user_input.text,
        )

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage2] Input='%s'. Entering Chat Mode.", user_input.text)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # Initialize history if new
        if key not in self._chat_sessions:
            self._chat_sessions[key] = []
            _LOGGER.debug("[Stage2] Starting new chat session for %s", key)

        history = self._chat_sessions[key]
        
        # Add User input to history (internal format)
        history.append({"role": "user", "content": user_input.text})

        # Generate Response via Gemini
        chat_cap = self.get("chat")
        result = await chat_cap.run(user_input, history=history)

        # Extract text for history update
        response_text = "..."
        if result and result.response and result.response.speech:
            response_text = result.response.speech.get("plain", {}).get("speech", "")
        
        # Add Assistant response to history
        history.append({"role": "assistant", "content": response_text})

        return {"status": "handled", "result": result}
