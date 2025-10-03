import logging
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation
from homeassistant.helpers import intent

from .prompt_executor import PromptExecutor
from .prompts import CLARIFICATION_PROMPT, CLARIFICATION_PROMPT_STAGE2
from .stage0 import Stage0Result

_LOGGER = logging.getLogger(__name__)


def _with_new_text(user_input: conversation.ConversationInput, new_text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=getattr(user_input, "satellite_id", None),
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
    )


class Stage1Processor:
    """Stage 1: Clarification (two-step)."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)

    async def _make_continuing_response(
        self, message: str, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        resp = intent.IntentResponse(language=user_input.language or "de")
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)
        _LOGGER.debug("Continuing response: %s", message)
        return conversation.ConversationResult(
            response=resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=True,
        )

    async def run(self, user_input: conversation.ConversationInput, raw_stage0=None):
        # Stage1-A
        _LOGGER.debug("Stage1 clarification for input: %s", user_input.text)
        data = await self.prompts.run(CLARIFICATION_PROMPT, {"user_input": user_input.text})
        _LOGGER.debug("Stage1 clarification result: %s", data)

import logging
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation
from homeassistant.helpers import intent

from .prompt_executor import PromptExecutor
from .prompts import CLARIFICATION_PROMPT, CLARIFICATION_PROMPT_STAGE2
from .stage0 import Stage0Result

_LOGGER = logging.getLogger(__name__)


def _with_new_text(user_input: conversation.ConversationInput, new_text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=getattr(user_input, "satellite_id", None),
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
    )


class Stage1Processor:
    """Stage 1: Clarification (two-step)."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)

    async def _make_continuing_response(
        self, message: str, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        resp = intent.IntentResponse(language=user_input.language or "de")
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)
        _LOGGER.debug("Continuing response: %s", message)
        return conversation.ConversationResult(
            response=resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=True,
        )

    async def run(self, user_input: conversation.ConversationInput, raw_stage0=None):
        # Stage1-A
        _LOGGER.debug("Stage1 clarification for input: %s", user_input.text)
        data = await self.prompts.run(CLARIFICATION_PROMPT, {"user_input": user_input.text})
        _LOGGER.debug("Stage1 clarification result: %s", data)

        # If Stage1-A can rewrite into explicit commands (list of strings), try them directly
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            # If it simply echoed the input, delegate to Stage1-B
            if len(data) == 1 and data[0].strip() == user_input.text.strip():
                _LOGGER.info("Stage1-A returned identical text → delegate to Stage1-B")
            else:
                results = []
                for clarified_command in data:
                    clarified_input = _with_new_text(user_input, clarified_command)
                    res = await conversation.async_converse(
                        self.hass,
                        text=clarified_input.text,
                        context=clarified_input.context,
                        conversation_id=clarified_input.conversation_id,
                        language=clarified_input.language or "de",
                        agent_id=conversation.HOME_ASSISTANT_AGENT,
                    )
                    results.append(res)
                # Return only the last result to keep conversation state sane
                return results[-1] if results else await self._make_continuing_response(
                    "Entschuldigung, ich konnte das nicht verarbeiten.", user_input
                )

        # Stage1-B (fallback to stronger clarifier)
        _LOGGER.debug("Stage1-A insufficient → Stage1-B")
        data2 = await self.prompts.run(CLARIFICATION_PROMPT_STAGE2, {"user_input": user_input.text})
        _LOGGER.debug("Stage1-B clarification result: %s", data2)

        if not isinstance(data2, dict):
            _LOGGER.error("Stage1-B returned invalid format: %s", data2)
            return await self._make_continuing_response(
                "Entschuldigung, ich konnte deine Anweisung nicht verstehen.", user_input
            )

        # Pass to DefaultAgent; Stage2 will pick it up later if needed
        return await conversation.async_converse(
            self.hass,
            text=user_input.text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )
