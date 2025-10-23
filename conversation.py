import logging
from typing import Any, List

from homeassistant.components import conversation

from .stage0 import Stage0Processor
from .stage1 import Stage1Processor
from .stage2 import Stage2Processor

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Dynamic N-stage orchestrator for Home Assistant Assist."""

    def __init__(self, hass, config):
        self.hass = hass
        self.hass.data["custom_components.multistage_assist_agent"] = self
        self.config = config
        self.stages: List[Any] = [
            Stage0Processor(hass, config),
            Stage1Processor(hass, config),
            Stage2Processor(hass, config),
        ]

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    async def _fallback(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Single place to hit the default HA agent."""
        return await conversation.async_converse(
            self.hass,
            text=user_input.text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        _LOGGER.info("Received utterance: %s", user_input.text)

        # If any stage owns a pending turn, let it resolve first.
        for stage in self.stages:
            if hasattr(stage, "has_pending") and stage.has_pending(user_input):
                _LOGGER.debug("Resuming pending interaction in %s", stage.__class__.__name__)
                pending = await stage.resolve_pending(user_input)
                if not pending:
                    _LOGGER.warning("%s returned None on pending resolution", stage.__class__.__name__)
                    break

                status, value = pending.get("status"), pending.get("result")
                if status == "handled":
                    return value or await self._fallback(user_input)
                if status == "error":
                    return value or await self._fallback(user_input)
                if status == "escalate":
                    return await self._run_pipeline(user_input, value)

                _LOGGER.warning("Unexpected pending format from %s: %s", stage.__class__.__name__, pending)

        # Fresh pipeline
        result = await self._run_pipeline(user_input)
        return result or await self._fallback(user_input)

    async def _run_pipeline(self, user_input: conversation.ConversationInput, prev_result: Any = None):
        """Run through the stages sequentially until one handles the input."""
        current = prev_result
        for stage in self.stages:
            try:
                out = await stage.run(user_input, current)
            except Exception:
                _LOGGER.exception("%s failed", stage.__class__.__name__)
                raise

            if not isinstance(out, dict):
                _LOGGER.warning("%s returned invalid result format: %s", stage.__class__.__name__, out)
                continue

            status, value = out.get("status"), out.get("result")
            if status == "handled":
                return value or None
            if status == "escalate":
                current = value
                continue
            if status == "error":
                return value or None

        _LOGGER.warning("All stages exhausted without a ConversationResult.")
        return None
