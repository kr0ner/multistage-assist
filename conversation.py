import logging
import json
from homeassistant.components import conversation
from hassil.recognize import recognize_best

from .prompt_executor import PromptExecutor
from .entity_resolver import EntityResolver
from .utils import make_response
from .prompts import (
    PLURAL_SINGULAR_PROMPT,
    DISAMBIGUATION_PROMPT,
    DISAMBIGUATION_RESOLUTION_PROMPT,
    CLARIFICATION_PROMPT,
)

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self.entities = EntityResolver(hass)
        self._pending_disambiguation: dict[str, dict[str, str]] = {}

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    async def _dry_run_recognize(self, utterance, language, user_input):
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, conversation.DefaultAgent):
            _LOGGER.warning("Only works with DefaultAgent right now")
            return None

        lang_intents = await agent.async_get_or_load_intents(language)
        if lang_intents is None:
            return None

        slot_lists = await agent._make_slot_lists()
        intent_context = agent._make_intent_context(user_input)

        def _run():
            return recognize_best(
                utterance,
                lang_intents.intents,
                slot_lists=slot_lists,
                intent_context=intent_context,
                language=language,
                best_metadata_key="hass_custom_sentence",
                best_slot_name="name",
            )

        return await self.hass.async_add_executor_job(_run)

    async def _is_plural(self, text: str) -> bool | None:
        context = {"user_input": text}
        data = await self.prompts.run(PLURAL_SINGULAR_PROMPT, context)
        return data and data.get("multiple_entities") == "true"

    async def _call_stage1_clarification(self, user_input, resp=None):
        context = {"user_input": user_input.text}
        data = await self.prompts.run(CLARIFICATION_PROMPT, context)
        return make_response(data["message"], user_input.language)

    async def _call_stage1_disambiguation(self, user_input, entity_ids: list[str]):
        entity_map = await self.entities.make_entity_map(entity_ids)
        if await self._is_plural(user_input.text):
            return await self._delegate_to_default_agent(user_input)

        context = {"user_input": user_input.text, "entities": entity_map}
        data = await self.prompts.run(DISAMBIGUATION_PROMPT, context)
        self._pending_disambiguation[user_input.conversation_id] = entity_map
        return make_response(data["message"], user_input.language)

    async def _resolve_disambiguation_answer(self, user_input, candidates: dict[str, str]):
        context = {"user_input": user_input.text, "entities": candidates}
        data = await self.prompts.run(
            DISAMBIGUATION_RESOLUTION_PROMPT,
            context,
            temperature=0.25,
        )

        if data.get("action") == "abort":
            return make_response("Okay, abgebrochen.", user_input.language)

        if not data.get("entities"):
            return await self._delegate_to_default_agent(user_input)

        return await self._delegate_to_default_agent(user_input)

    async def _delegate_to_default_agent(self, user_input):
        return await conversation.async_converse(
            self.hass,
            text=user_input.text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        utterance = user_input.text
        language = user_input.language or "de"
        _LOGGER.info("Received utterance: %s", utterance)

        try:
            result = await self._dry_run_recognize(utterance, language, user_input)
            if not result or not result.intent:
                return await self._call_stage1_clarification(user_input)

            entities = {k: v.value for k, v in (result.entities or {}).items()}
            resolved = await self.entities.resolve(entities)

            if not resolved.merged:
                return await self._call_stage1_clarification(user_input, result)
            if len(resolved.merged) > 1:
                return await self._call_stage1_disambiguation(user_input, resolved.merged)

            return await self._delegate_to_default_agent(user_input)

        except Exception as err:
            _LOGGER.warning("Stage 0 failed: %s", err)
            # If dry-run blows up completely, fallback to clarification path
            return await self._call_stage1_clarification(user_input)
