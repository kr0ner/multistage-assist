import logging
from typing import Dict, Any

from homeassistant.components import conversation
from homeassistant.components.conversation.default_agent import DefaultAgent
from hassil.recognize import recognize_best
from homeassistant.helpers import intent

from .base_stage import BaseStage
from .capabilities.entity_resolver import EntityResolverCapability
from .conversation_utils import error_response
from .stage_result import Stage0Result

_LOGGER = logging.getLogger(__name__)


class Stage0Processor(BaseStage):
    """Stage 0: Dry-run NLU and early entity resolution (no LLM)."""
    name = "stage0"

    async def _dry_run_recognize(self, user_input: conversation.ConversationInput):
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, DefaultAgent):
            _LOGGER.debug("[Stage0] DefaultAgent not available → skipping dry-run recognize.")
            return None

        language = user_input.language or "de"
        lang_intents = await agent.async_get_or_load_intents(language)
        if not lang_intents:
            _LOGGER.debug("[Stage0] No language intents loaded for '%s'.", language)
            return None

        slot_lists = await agent._make_slot_lists()
        intent_context = agent._make_intent_context(user_input)

        def _run():
            return recognize_best(
                user_input.text,
                lang_intents.intents,
                slot_lists=slot_lists,
                intent_context=intent_context,
                language=language,
                best_metadata_key="hass_custom_sentence",
                best_slot_name="name",
            )

        return await self.hass.async_add_executor_job(_run)

    def _normalize_entities(self, entities: Dict[str, Any] | None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not entities:
            return out
        for k, v in entities.items():
            out[str(k)] = getattr(v, "value", v)
        return out

    async def run(self, user_input: conversation.ConversationInput, prev_result=None):
        _LOGGER.debug(
            "[Stage0] Input='%s', prev_result=%s",
            user_input.text,
            type(prev_result).__name__,
        )

        match = await self._dry_run_recognize(user_input)
        if not match or not getattr(match, "intent", None):
            _LOGGER.debug("[Stage0] No NLU match → escalate.")
            return {"status": "escalate", "result": None}

        intent_name = getattr(match.intent, "name", None) or match.intent
        _LOGGER.debug("[Stage0] NLU matched intent='%s'", intent_name)

        norm_entities = self._normalize_entities(getattr(match, "entities", None))
        if norm_entities:
            _LOGGER.debug("[Stage0] NLU extracted entities (raw) keys=%s", list(norm_entities.keys()))
        else:
            _LOGGER.debug("[Stage0] NLU extracted no entities")

        # Resolve entities (map friendly/area/slots to concrete entity_ids)
        resolver = EntityResolverCapability(self.hass, self.config)
        resolved = await resolver.run(user_input, entities=norm_entities)
        resolved_ids = (resolved or {}).get("resolved_ids", [])
        _LOGGER.debug("[Stage0] Entity resolver returned %d id(s): %s", len(resolved_ids), resolved_ids)

        # Prepare Stage0Result once
        result = Stage0Result(
            type=("intent" if resolved_ids else "clarification"),
            intent=intent_name,
            raw=user_input.text,
            resolved_ids=resolved_ids,
        )

        # Too many candidates? Ask next stage to clarify.
        threshold = int(getattr(self.config, "early_filter_threshold", 10))
        if resolved_ids and len(resolved_ids) > threshold:
            _LOGGER.debug(
                "[Stage0] %d candidates exceed threshold=%d → escalate for clarification.",
                len(resolved_ids),
                threshold,
            )
            result = Stage0Result(
                type="clarification",
                intent=intent_name,
                raw=user_input.text,
                resolved_ids=resolved_ids,
            )
            return {"status": "escalate", "result": result}

        # If nothing concrete yet → escalate
        if not resolved_ids:
            _LOGGER.debug("[Stage0] No concrete targets resolved → escalate.")
            return {"status": "escalate", "result": result}

        # Single, known Hass intent → execute directly through HA agent
        if len(resolved_ids) == 1 and intent_name and intent_name.startswith("Hass"):
            try:
                _LOGGER.debug(
                    "[Stage0] Direct execution path: intent='%s', single target=%s → delegating to HA agent.",
                    intent_name,
                    resolved_ids[0],
                )
                agent = conversation.async_get_agent(self.hass)
                exec_result = await agent.async_handle_intents(user_input)

                if isinstance(exec_result, conversation.ConversationResult):
                    _LOGGER.debug("[Stage0] Agent returned ConversationResult → handled.")
                    return {"status": "handled", "result": exec_result}

                if isinstance(exec_result, intent.IntentResponse):
                    _LOGGER.debug("[Stage0] Agent returned IntentResponse → wrapped as ConversationResult.")
                    conv_result = conversation.ConversationResult(
                        response=exec_result,
                        conversation_id=user_input.conversation_id,
                        continue_conversation=True,
                    )
                    return {"status": "handled", "result": conv_result}

                _LOGGER.error("[Stage0] Unexpected response type from agent: %r", type(exec_result))
                return {
                    "status": "error",
                    "result": await error_response(
                        user_input,
                        "Fehler: Unerwarteter Antworttyp vom Intent-Handler.",
                    ),
                }
            except Exception as err:
                _LOGGER.exception("[Stage0] Intent execution crashed: %s", err)
                return {
                    "status": "error",
                    "result": await error_response(user_input, f"Interner Fehler beim Ausführen: {err}"),
                }

        # Multiple candidates → escalate for disambiguation
        _LOGGER.debug(
            "[Stage0] %d candidate(s) and intent='%s' → escalate to Stage1 for disambiguation.",
            len(resolved_ids),
            intent_name,
        )
        return {"status": "escalate", "result": result}
