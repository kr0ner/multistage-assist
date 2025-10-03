import logging
from typing import Dict, Any

from homeassistant.components import conversation

from .stage0 import Stage0Processor, Stage0Result
from .stage1 import Stage1Processor
from .stage2 import Stage2Processor

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant (orchestrator)."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config

        self.stage0 = Stage0Processor(hass, config)
        self.stage1 = Stage1Processor(hass, config)
        self.stage2 = Stage2Processor(hass, config)

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        utterance = user_input.text
        _LOGGER.info("Received utterance: %s", utterance)

        # If there is a pending disambiguation/selection workflow in Stage2, resolve it first.
        if self.stage2.has_pending(user_input):
            return await self.stage2.resolve_pending(user_input)

        # Stage 0: NLU + entity resolution + early filtering
        s0: Stage0Result | None = await self.stage0.run(user_input)

        if s0 is None:
            # No intent → ask Stage1 to clarify
            return await self.stage1.run(user_input)

        if s0.type == "clarification":
            # NLU found something but no usable entities → Stage1 clarification
            return await self.stage1.run(user_input, s0.raw)

        if s0.type == "intent":
            # We have an intent; Stage2 executes (get_value, control, disambiguation etc.)
            return await self.stage2.run(user_input, s0)

        # Fallback: clarify
        _LOGGER.warning("Unknown Stage0 result type: %s", getattr(s0, "type", None))
        return await self.stage1.run(user_input)
