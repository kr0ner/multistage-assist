import logging
import json
from typing import Any, Dict, List

from homeassistant.components import conversation
from homeassistant.helpers import intent
from hassil.recognize import recognize_best

from .prompt_executor import PromptExecutor
from .entity_resolver import EntityResolver
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
        # session or conversation → state
        self._pending_disambiguation: Dict[str, Dict[str, Any]] = {}

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    def _get_state_key(self, user_input) -> str:
        # Prefer session_id if available, otherwise conversation_id
        return getattr(user_input, "session_id", None) or user_input.conversation_id

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
        is_plural = bool(data and data.get("multiple_entities") == "true")
        _LOGGER.debug("Plural detection for '%s' -> %s (raw=%s)", text, is_plural, data)
        return is_plural

    async def _make_continuing_response(
        self, message: str, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Return a ConversationResult that keeps the session open."""
        resp = intent.IntentResponse(language=user_input.language or "de")
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)

        return conversation.ConversationResult(
            response=resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=True,
        )

    async def _call_stage1_clarification(self, user_input, resp=None):
        context = {"user_input": user_input.text}
        data = await self.prompts.run(CLARIFICATION_PROMPT, context)
        _LOGGER.debug("Clarification result for '%s': %s", user_input.text, data)

        if isinstance(data, dict) and "message" in data:
            return await self._make_continuing_response(data["message"], user_input)

        return await self._delegate_to_default_agent(user_input)

    async def _call_stage1_disambiguation(self, user_input, entity_ids: List[str], intent=None):
        entity_map = await self.entities.make_entity_map(entity_ids)
        _LOGGER.debug("Disambiguation candidates (ids=%s) -> map=%s", entity_ids, entity_map)

        if await self._is_plural(user_input.text):
            _LOGGER.info("Plural detected; delegating to default agent with original utterance.")
            return await self._delegate_to_default_agent(user_input)

        context = {"user_input": user_input.text, "entities": entity_map}
        data = await self.prompts.run(DISAMBIGUATION_PROMPT, context)
        _LOGGER.debug("Disambiguation prompt output: %s", data)

        key = self._get_state_key(user_input)
        self._pending_disambiguation[key] = {
            "intent": intent,
            "candidates": entity_map,
            "order": entity_ids,
        }
        _LOGGER.debug(
            "Saved pending disambiguation state for %s: %s",
            key,
            self._pending_disambiguation[key],
        )

        msg = (data or {}).get("message") or "Bitte präzisiere, welches Gerät du meinst."
        return await self._make_continuing_response(msg, user_input)

    async def _resolve_disambiguation_answer(self, user_input):
        key = self._get_state_key(user_input)
        pending = self._pending_disambiguation.get(key)
        _LOGGER.debug("Resolving disambiguation for %s: pending=%s", key, pending)

        if not pending:
            _LOGGER.info("No pending disambiguation for %s; delegating.", key)
            return await self._delegate_to_default_agent(user_input)

        context = {
            "user_input": user_input.text,
            "entities": pending["candidates"],
            "order": pending.get("order", []),
        }
        _LOGGER.debug("Resolution context: %s", context)

        data = await self.prompts.run(
            DISAMBIGUATION_RESOLUTION_PROMPT,
            context,
            temperature=0.25,
        )
        _LOGGER.debug("Resolution LLM output for '%s': %s", user_input.text, data)

        entities = (data or {}).get("entities") or []
        message = (data or {}).get("message") or ""
        action = (data or {}).get("action")

        if action == "abort":
            _LOGGER.info("User aborted disambiguation for %s", key)
            self._pending_disambiguation.pop(key, None)
            resp = intent.IntentResponse(language=user_input.language or "de")
            resp.response_type = intent.IntentResponseType.QUERY_ANSWER
            resp.async_set_speech("Okay, abgebrochen.")
            return conversation.ConversationResult(
                response=resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )

        if not entities:
            _LOGGER.warning(
                "Disambiguation failed for '%s' (data=%s). Keeping state for retry.",
                user_input.text,
                data,
            )
            return await self._make_continuing_response(
                "Entschuldigung, ich habe das nicht verstanden. Welches Gerät meinst du?",
                user_input,
            )

        # ✅ Success: clear state
        self._pending_disambiguation.pop(key, None)

        intent_obj = pending.get("intent")
        _LOGGER.debug("Pending intent object before patching: %r", intent_obj)
        if intent_obj:
            try:
                intent_name = getattr(intent_obj, "name", None)

                # ✅ Build slots in HA format
                if len(entities) == 1:
                    slots: Dict[str, Dict[str, Any]] = {"name": {"value": entities[0]}}
                else:
                    slots = {"name": {"value": entities}}

                # Drop broad conflicts
                for conflict in ("domain", "area", "floor"):
                    if conflict in slots:
                        _LOGGER.debug("Removing conflicting slot: %s", conflict)
                        del slots[conflict]

                _LOGGER.debug(
                    "Executing patched intent '%s' with slots=%s",
                    intent_name,
                    slots,
                )

                resp = await intent.async_handle(
                    self.hass,
                    platform="conversation",
                    intent_type=intent_name,
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context,
                    language=user_input.language or "de",
                )

                # ✅ Always ensure speech is set
                if not resp.speech:
                    if message:
                        resp.async_set_speech(message)
                    else:
                        # fallback if even LLM gave no message
                        if len(entities) == 1:
                            pretty = pending["candidates"].get(entities[0], entities[0])
                            resp.async_set_speech(f"Okay, ich schalte {pretty} ein.")
                        else:
                            resp.async_set_speech("Okay, ich habe die Geräte geschaltet.")

                return conversation.ConversationResult(
                    response=resp,
                    conversation_id=user_input.conversation_id,
                    continue_conversation=False,
                )
            except Exception as e:
                _LOGGER.error("Failed to patch/execute intent_obj=%r error=%s", intent_obj, e)

        _LOGGER.warning(
            "Could not patch disambiguated intent. Falling back. entities=%s intent_obj=%r",
            entities,
            intent_obj,
        )
        return await self._make_continuing_response(
            "Entschuldigung, das konnte ich nicht ausführen.", user_input
        )

    async def _delegate_to_default_agent(self, user_input):
        _LOGGER.debug("Delegating to Home Assistant Default Agent for utterance: %s", user_input.text)
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

        key = self._get_state_key(user_input)

        if key in self._pending_disambiguation:
            _LOGGER.debug("Conversation %s is in disambiguation mode.", key)
            return await self._resolve_disambiguation_answer(user_input)

        try:
            result = await self._dry_run_recognize(utterance, language, user_input)
            if not result or not result.intent:
                _LOGGER.debug("NLU did not produce an intent; entering clarification.")
                return await self._call_stage1_clarification(user_input)

            entities = {k: v.value for k, v in (result.entities or {}).items()}
            _LOGGER.debug("NLU extracted entities: %s", entities)
            resolved = await self.entities.resolve(entities)
            _LOGGER.debug(
                "Resolved entity_ids: by_area=%s, by_name=%s, merged=%s",
                resolved.by_area,
                resolved.by_name,
                resolved.merged,
            )

            if not resolved.merged:
                _LOGGER.debug("No entity_ids resolved; entering clarification.")
                return await self._call_stage1_clarification(user_input, result)

            if len(resolved.merged) > 1:
                _LOGGER.debug(
                    "Multiple entity_ids resolved; entering disambiguation for %s", resolved.merged
                )
                return await self._call_stage1_disambiguation(
                    user_input, resolved.merged, intent=result.intent
                )

            _LOGGER.debug("Single entity resolved; delegating to default agent for execution.")
            return await self._delegate_to_default_agent(user_input)

        except Exception as err:
            _LOGGER.warning("Stage 0 failed: %s", err)
            return await self._call_stage1_clarification(user_input)
