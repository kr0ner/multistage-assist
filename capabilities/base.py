import logging
from typing import Any, Dict, Optional
from homeassistant.components import conversation

_LOGGER = logging.getLogger(__name__)


class Capability:
    """Base class for a reusable reasoning or execution skill."""

    name: str = "generic"
    description: str = ""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config

    async def run(
        self,
        user_input: conversation.ConversationInput,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute capability logic. Must be overridden by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    async def _safe_prompt(
        self,
        prompt_def: Dict[str, Any],
        variables: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Convenience helper to run the shared PromptExecutor."""
        from ..prompt_executor import PromptExecutor

        executor = PromptExecutor(self.config)
        try:
            _LOGGER.debug(
                "[Capability:%s] Executing prompt with vars=%s",
                self.name,
                list(variables.keys()),
            )
            data = await executor.run(prompt_def, variables, temperature=temperature)
            _LOGGER.debug("[Capability:%s] Prompt result=%s", self.name, data)
            return data
        except Exception:
            _LOGGER.exception("[Capability:%s] Prompt execution failed", self.name)
            return None

    async def _execute_intent(
        self,
        user_input: conversation.ConversationInput,
        text: str,
        language: Optional[str] = None,
    ) -> conversation.ConversationResult:
        """Shortcut to run a Home Assistant intent execution."""
        _LOGGER.debug("[Capability:%s] Executing HA intent: %s", self.name, text)
        return await conversation.async_converse(
            self.hass,
            text=text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=language or user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )

    # Optional helpers that subclasses can override
    async def prepare_context(self, user_input: conversation.ConversationInput) -> Dict[str, Any]:
        """Extract structured context for the capability."""
        return {}
