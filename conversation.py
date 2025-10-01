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
    ENTITY_FILTER_PROMPT,
    DISAMBIGUATION_PROMPT,
    DISAMBIGUATION_RESOLUTION_PROMPT,
    CLARIFICATION_PROMPT,
    GET_VALUE_PHRASE_PROMPT,
    CLARIFICATION_PROMPT_STAGE2,
)

_LOGGER = logging.getLogger(__name__)


def _with_new_text(
    user_input: conversation.ConversationInput, new_text: str
) -> conversation.ConversationInput:
    """Return a new ConversationInput with modified text but same metadata."""
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        language=user_input.language,
        agent_id=user_input.agent_id,
        device_id=user_input.device_id,
    )

class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Multi-Stage Assist Agent for Home Assistant."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self.entities = EntityResolver(hass)
        self._pending_disambiguation: Dict[str, Dict[str, Any]] = {}

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    def _get_state_key(self, user_input) -> str:
        return getattr(user_input, "session_id", None) or user_input.conversation_id

    async def _dry_run_recognize(self, utterance, language, user_input):
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, conversation.DefaultAgent):
            _LOGGER.warning("Only works with DefaultAgent right now")
            return None

        lang_intents = await agent.async_get_or_load_intents(language)
        if lang_intents is None:
            _LOGGER.debug("No intents loaded for language=%s", language)
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

        _LOGGER.debug("Running dry-run recognize for utterance='%s'", utterance)
        return await self.hass.async_add_executor_job(_run)

    async def _is_plural(self, text: str) -> bool | None:
        context = {"user_input": text}
        data = await self.prompts.run(PLURAL_SINGULAR_PROMPT, context)
        is_plural = bool(data and data.get("multiple_entities"))
        _LOGGER.debug("Plural detection for '%s' -> %s (raw=%s)", text, is_plural, data)
        return is_plural

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

    async def _handle_get_value(
        self, eid: str, data: dict[str, Any], user_input
    ) -> conversation.ConversationResult:
        _LOGGER.debug("Handling get_value for entity=%s with data=%s", eid, data)
        state = self.hass.states.get(eid)
        if not state or state.state in ("unknown", "unavailable", None):
            _LOGGER.warning("Entity %s unavailable or unknown", eid)
            return await self._make_continuing_response(
                "Der Wert ist derzeit nicht verfügbar.", user_input
            )

        context = {
            "measurement": data.get("measurement") or "value",
            "value": state.state,
            "unit": state.attributes.get("unit_of_measurement") or "",
            "area": data.get("area"),
        }
        _LOGGER.debug("Value context for phrasing: %s", context)

        phrased = await self.prompts.run(GET_VALUE_PHRASE_PROMPT, context)
        _LOGGER.debug("Value phrasing output: %s", phrased)
        message = (phrased or {}).get("message") or f"{context['measurement']}: {context['value']} {context['unit']}"

        return await self._make_continuing_response(message, user_input)

    # -------------------------------------------------------------------------
    # Clarification
    # -------------------------------------------------------------------------

    async def _call_stage1_clarification(self, user_input, resp=None):
        _LOGGER.debug("Stage1 clarification for input: %s", user_input.text)
        context = {"user_input": user_input.text}
        data = await self.prompts.run(CLARIFICATION_PROMPT, context)
        _LOGGER.debug("Stage1 clarification result: %s", data)

        if isinstance(data, dict) and "message" in data:
            _LOGGER.info("Stage1 clarification returned fallback dict → escalating to Stage2")
            return await self._call_stage2_clarification(user_input)

        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            if len(data) == 1 and data[0].strip() == user_input.text.strip():
                _LOGGER.info("Stage1 clarification returned same text → escalating to Stage2")
                return await self._call_stage2_clarification(user_input)

            results = []
            for clarified_command in data:
                _LOGGER.debug("Processing clarified command: %s", clarified_command)
                clarified_input = _with_new_text(user_input, clarified_command)
                result = await self._delegate_to_default_agent(clarified_input)
                results.append(result)
            return results

        _LOGGER.warning("Stage1 clarification invalid → escalating to Stage2")
        return await self._call_stage2_clarification(user_input)

    # -------------------------------------------------------------------------
    # Disambiguation: Action Intents
    # -------------------------------------------------------------------------

    async def _call_action_disambiguation(self, user_input, entity_ids: List[str], intent=None):
        entity_map = await self.entities.make_entity_map(entity_ids)
        _LOGGER.debug("Action disambiguation candidates (ids=%s) -> %s", entity_ids, entity_map)

        if await self._is_plural(user_input.text):
            _LOGGER.info("Plural detected, delegating to default agent")
            return await self._delegate_to_default_agent(user_input)

        context = {"input_entities": entity_map}
        data = await self.prompts.run(DISAMBIGUATION_PROMPT, context)
        _LOGGER.debug("Action disambiguation prompt output: %s", data)

        key = self._get_state_key(user_input)
        self._pending_disambiguation[key] = {
            "type": "action",
            "intent": intent,
            "candidates": entity_map,
        }
        _LOGGER.debug("Saved pending action disambiguation for %s", key)

        msg = (data or {}).get("message") or "Bitte präzisiere, welches Gerät du meinst."
        return await self._make_continuing_response(msg, user_input)

    async def _resolve_action_disambiguation(self, user_input, pending, entities, message):
        intent_obj = pending.get("intent")
        if not intent_obj:
            _LOGGER.warning("No intent object for action disambiguation")
            return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

        try:
            intent_name = getattr(intent_obj, "name", None)
            responses = []
            for eid in entities:
                slots: Dict[str, Any] = {"name": {"value": eid}}
                _LOGGER.debug("Executing action intent '%s' with slots=%s", intent_name, slots)
                resp = await intent.async_handle(
                    self.hass,
                    platform="conversation",
                    intent_type=intent_name,
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context,
                    language=user_input.language or "de",
                )
                responses.append(resp)

            final_resp = responses[-1]
            if not final_resp.speech:
                if message:
                    final_resp.async_set_speech(message)
                else:
                    pretty = ", ".join(pending["candidates"].get(eid, eid) for eid in entities)
                    final_resp.async_set_speech(f"Okay, ich schalte {pretty} ein.")

            _LOGGER.debug("Action disambiguation resolved successfully")
            return conversation.ConversationResult(
                response=final_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
        except Exception as e:
            _LOGGER.error("Failed to execute action disambiguation: %s", e)
            return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

    # -------------------------------------------------------------------------
    # Disambiguation: Value (get_value)
    # -------------------------------------------------------------------------

    async def _call_value_disambiguation(self, user_input, entity_ids: List[str], clarification_data: dict[str, Any]):
        entity_map = await self.entities.make_entity_map(entity_ids)
        _LOGGER.debug("Value disambiguation candidates (ids=%s) -> %s", entity_ids, entity_map)

        context = {"input_entities": entity_map}
        data = await self.prompts.run(DISAMBIGUATION_PROMPT, context)
        _LOGGER.debug("Value disambiguation prompt output: %s", data)

        key = self._get_state_key(user_input)
        self._pending_disambiguation[key] = {
            "type": "value",
            "clarification_data": clarification_data,
            "candidates": entity_map,
        }
        _LOGGER.debug("Saved pending value disambiguation for %s", key)

        msg = (data or {}).get("message") or "Bitte präzisiere, welchen Sensor du meinst."
        return await self._make_continuing_response(msg, user_input)

    async def _resolve_value_disambiguation(self, user_input, pending, entities):
        _LOGGER.debug("Resolving value disambiguation with entities=%s", entities)
        clarification_data = pending.get("clarification_data")
        eid = entities[0] if len(entities) == 1 else None
        if eid:
            return await self._handle_get_value(eid, clarification_data, user_input)
        _LOGGER.info("Multiple sensors still unresolved, restarting value disambiguation")
        return await self._call_value_disambiguation(user_input, entities, clarification_data)

    # -------------------------------------------------------------------------
    # Disambiguation Resolver
    # -------------------------------------------------------------------------

    async def _resolve_disambiguation_answer(self, user_input):
        key = self._get_state_key(user_input)
        pending = self._pending_disambiguation.get(key)
        _LOGGER.debug("Resolving disambiguation for %s: %s", key, pending)

        if not pending:
            _LOGGER.info("No pending disambiguation found, delegating to default agent")
            return await self._delegate_to_default_agent(user_input)

        context = {"user_input": user_input.text, "input_entities": pending["candidates"]}
        data = await self.prompts.run(DISAMBIGUATION_RESOLUTION_PROMPT, context, temperature=0.25)
        _LOGGER.debug("Disambiguation resolution output: %s", data)

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
            _LOGGER.warning("No entities resolved in disambiguation")
            return await self._make_continuing_response(
                "Entschuldigung, ich habe das nicht verstanden. Bitte wiederhole.", user_input
            )

        self._pending_disambiguation.pop(key, None)

        if pending["type"] == "value":
            return await self._resolve_value_disambiguation(user_input, pending, entities)

        if pending["type"] == "action":
            return await self._resolve_action_disambiguation(user_input, pending, entities, message)

        _LOGGER.error("Unknown disambiguation type: %s", pending.get("type"))
        return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

    async def _call_stage2_clarification(self, user_input):
        _LOGGER.debug("Stage2 clarification for input: %s", user_input.text)
        context = {"user_input": user_input.text}
        data = await self.prompts.run(CLARIFICATION_PROMPT_STAGE2, context)
        _LOGGER.debug("Stage2 clarification result: %s", data)

        if not isinstance(data, dict):
            _LOGGER.error("Stage2 returned invalid format: %s", data)
            return await self._make_continuing_response(
                "Entschuldigung, ich konnte deine Anweisung nicht verstehen.", user_input
            )

        intention = data.get("intention")
        entities = await self.entities.resolve(data)
        _LOGGER.debug("Stage2 resolved entities: %s", entities.merged)

        if not entities.merged:
            _LOGGER.warning("Stage2 clarification found no entities")
            return await self._make_continuing_response(
                "Ich konnte kein passendes Gerät finden.", user_input
            )

        # --- Special case: get_value ---
        if intention == "HassGetState":
            all_sensors = [
                eid for eid in entities.merged
                if eid.startswith("sensor.")
                and self.hass.states.get(eid) is not None
            ]
            _LOGGER.debug("Candidate sensors in area: %s", all_sensors)

            keyword = (data.get("measurement") or "").lower()
            filtered = []
            if keyword:
                for eid in all_sensors:
                    state = self.hass.states.get(eid)
                    fname = state.attributes.get("friendly_name", "").lower()
                    if keyword in eid.lower() or keyword in fname:
                        filtered.append(eid)

            if not filtered:
                _LOGGER.warning(
                    "No matching sensors found for measurement='%s' in candidates=%s",
                    keyword, all_sensors
                )
                return await self._make_continuing_response(
                    f"Ich konnte keinen passenden Sensor für {keyword or 'diesen Wert'} finden.",
                    user_input,
                )

            if len(filtered) == 1:
                return await self._handle_get_value(filtered[0], data, user_input)

            return await self._call_value_disambiguation(user_input, filtered, data)

        # --- Regular control intents ---
        if len(entities.merged) == 1:
            eid = entities.merged[0]
            intent_name = intention
            slots: Dict[str, Any] = {"name": {"value": eid}}
            _LOGGER.debug(
                "Stage2 clarification resolved: executing intent '%s' on %s",
                intent_name, eid
            )
            try:
                resp = await intent.async_handle(
                    self.hass,
                    platform="conversation",
                    intent_type=intent_name,
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context,
                    language=user_input.language or "de",
                )

                # --- Ensure speech is always set ---
                if not resp.speech:
                    pretty = self.hass.states.get(eid).attributes.get("friendly_name", eid)
                    if intent_name == "HassTurnOn":
                        resp.async_set_speech(f"Okay, {pretty} ist jetzt eingeschaltet.")
                    elif intent_name == "HassTurnOff":
                        resp.async_set_speech(f"Okay, {pretty} ist jetzt ausgeschaltet.")
                    else:
                        resp.async_set_speech("Okay, erledigt.")

                return conversation.ConversationResult(
                    response=resp,
                    conversation_id=user_input.conversation_id,
                    continue_conversation=False,
                )
            except Exception as e:
                _LOGGER.error(
                    "Failed to execute intent '%s' for %s: %s", intent_name, eid, e
                )
                return await self._make_continuing_response(
                    "Entschuldigung, das konnte ich nicht ausführen.", user_input
                )

        _LOGGER.debug(
            "Stage2 clarification resulted in multiple entities, entering action disambiguation"
        )
        return await self._call_action_disambiguation(user_input, entities.merged)

    # -------------------------------------------------------------------------
    # Delegation + Process
    # -------------------------------------------------------------------------

    async def _delegate_to_default_agent(self, user_input):
        _LOGGER.debug("Delegating utterance='%s' to Home Assistant Default Agent", user_input.text)
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
            _LOGGER.debug("Conversation %s is in disambiguation mode", key)
            return await self._resolve_disambiguation_answer(user_input)

        try:
            result = await self._dry_run_recognize(utterance, language, user_input)
            if not result or not result.intent:
                _LOGGER.debug("NLU did not produce an intent, entering clarification")
                return await self._call_stage1_clarification(user_input)

            entities = {k: v.value for k, v in (result.entities or {}).items()}
            _LOGGER.debug("NLU extracted entities: %s", entities)
            resolved = await self.entities.resolve(entities)
            _LOGGER.debug("Resolved entity_ids: by_area=%s, by_name=%s, merged=%s",
                          resolved.by_area, resolved.by_name, resolved.merged)

            if not resolved.merged:
                _LOGGER.debug("No entities resolved, entering clarification")
                return await self._call_stage1_clarification(user_input, result)

            if len(resolved.merged) > 1:
                _LOGGER.debug("Multiple entities resolved, entering action disambiguation")
                return await self._call_action_disambiguation(user_input, resolved.merged, intent=result.intent)

            _LOGGER.debug("Single entity resolved, delegating to default agent")
            return await self._delegate_to_default_agent(user_input)

        except Exception as err:
            _LOGGER.warning("Stage 0 failed: %s", err)
            return await self._call_stage1_clarification(user_input)
