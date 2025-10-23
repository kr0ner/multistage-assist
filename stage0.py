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
            return None

        language = user_input.language or "de"
        lang_intents = await agent.async_get_or_load_intents(language)
        if not lang_intents:
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
        match = await self._dry_run_recognize(user_input)
        if not match or not getattr(match, "intent", None):
            return {"status": "escalate", "result": None}

        norm_entities = self._normalize_entities(getattr(match, "entities", None))

        # Resolve entities
        resolver = EntityResolverCapability(self.hass, self.config)
        resolved = await resolver.run(user_input, entities=norm_entities)
        resolved_ids = (resolved or {}).get("resolved_ids", [])

        # Prepare Stage0Result once
        intent_name = getattr(match.intent, "name", None) or match.intent
        result = Stage0Result(
            type=("intent" if resolved_ids else "clarification"),
            intent=intent_name,
            raw=user_input.text,
            resolved_ids=resolved_ids,
        )

        # Too many candidates? Ask next stage to clarify.
        threshold = int(getattr(self.config, "early_filter_threshold", 10))
        if resolved_ids and len(resolved_ids) > threshold:
            result = Stage0Result(
                type="clarification",
                intent=intent_name,
                raw=user_input.text,
                resolved_ids=resolved_ids,
            )
            return {"status": "escalate", "result": result}

        # If there is nothing concrete yet → escalate
        if not resolved_ids:
            return {"status": "escalate", "result": result}

        # Single, known Hass intent → execute directly
        if len(resolved_ids) == 1 and intent_name and intent_name.startswith("Hass"):
            try:
                agent = conversation.async_get_agent(self.hass)
                exec_result = await agent.async_handle_intents(user_input)

                if isinstance(exec_result, conversation.ConversationResult):
                    return {"status": "handled", "result": exec_result}

                if isinstance(exec_result, intent.IntentResponse):
                    conv_result = conversation.ConversationResult(
                        response=exec_result,
                        conversation_id=user_input.conversation_id,
                        continue_conversation=True,
                    )
                    return {"status": "handled", "result": conv_result}

                # Unknown type from agent
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
        return {"status": "escalate", "result": result}
