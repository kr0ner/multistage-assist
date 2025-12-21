"""Stage 0: NLU-based intent recognition and entity resolution.

Stage0 uses Home Assistant's built-in NLU (hassil) to recognize intents
without any LLM calls. It's the fastest and most reliable stage for
simple commands that match HA's sentence templates.

Flow:
1. Dry-run NLU recognition to get intent + entities
2. Memory alias resolution (TODO: integrate)
3. Entity resolution via EntityResolverCapability
4. Return StageResult.success if resolved, otherwise escalate
"""

import logging
from typing import Any, Dict, Optional

from homeassistant.components import conversation
from homeassistant.components.conversation.default_agent import DefaultAgent
from hassil.recognize import recognize_best

from .base_stage import BaseStage
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.intent_executor import IntentExecutorCapability
from .conversation_utils import error_response
from .stage_result import StageResult, Stage0Result
from .execution_pipeline import get_execution_pipeline

_LOGGER = logging.getLogger(__name__)


class Stage0Processor(BaseStage):
    """Stage 0: Dry-run NLU and early entity resolution (no LLM)."""

    name = "stage0"

    # Mapping specific HA intents to implied domains/device_classes
    INTENT_IMPLICATIONS = {
        "HassClimateGetTemperature": {"device_class": "temperature"},
        "HassTurnOn": {},
        "HassTurnOff": {},
        "HassLightSet": {"domain": "light"},
    }
    
    # Keys that are for resolution only, not execution params
    RESOLUTION_KEYS = {
        "area", "room", "floor", "name", "entity",
        "device", "label", "domain", "device_class", "entity_id",
    }

    async def _dry_run_recognize(self, user_input: conversation.ConversationInput):
        """Run NLU recognition without executing the intent."""
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
        """Normalize entity values from NLU match."""
        out: Dict[str, Any] = {}
        if not entities:
            return out
        for k, v in entities.items():
            out[str(k)] = getattr(v, "value", v)
        return out
    
    def _extract_params(self, norm_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract execution params, excluding resolution-only keys."""
        return {
            k: v for k, v in norm_entities.items() 
            if k not in self.RESOLUTION_KEYS
        }

    async def process(
        self, 
        user_input: conversation.ConversationInput, 
        context: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        """Process user input using NLU recognition.
        
        Args:
            user_input: ConversationInput from Home Assistant
            context: Optional context from previous stage (not used in Stage0)
            
        Returns:
            StageResult with status indicating outcome
        """
        context = context or {}
        
        _LOGGER.debug("[Stage0] Input='%s'", user_input.text)

        # 1. Dry-run NLU recognition
        match = await self._dry_run_recognize(user_input)
        if not match or not getattr(match, "intent", None):
            _LOGGER.debug("[Stage0] No NLU match → escalate.")
            return StageResult.escalate(
                context={"nlu_failed": True},
                raw_text=user_input.text,
            )

        intent_name = getattr(match.intent, "name", None) or match.intent
        _LOGGER.debug("[Stage0] NLU matched intent='%s'", intent_name)

        # 2. Normalize and enrich entities
        norm_entities = self._normalize_entities(getattr(match, "entities", None))
        
        # Inject implied constraints based on intent
        implications = self.INTENT_IMPLICATIONS.get(intent_name, {})
        if implications:
            _LOGGER.debug("[Stage0] Injecting constraints: %s", implications)
            norm_entities.update(implications)

        if norm_entities:
            _LOGGER.debug("[Stage0] NLU entities keys=%s", list(norm_entities.keys()))

        # 3. Entity resolution
        resolver = EntityResolverCapability(self.hass, self.config)
        resolved = await resolver.run(user_input, entities=norm_entities)
        resolved_ids = (resolved or {}).get("resolved_ids", [])
        
        _LOGGER.debug(
            "[Stage0] Entity resolver returned %d id(s): %s",
            len(resolved_ids), resolved_ids
        )

        # Build enriched context for next stage
        enriched_context = {
            "nlu_intent": intent_name,
            "nlu_entities": norm_entities,
            "resolved_ids": resolved_ids,
        }

        # 4. Check thresholds and decide action
        threshold = int(getattr(self.config, "early_filter_threshold", 10))
        
        if resolved_ids and len(resolved_ids) > threshold:
            _LOGGER.debug(
                "[Stage0] %d candidates exceed threshold=%d → escalate.",
                len(resolved_ids), threshold
            )
            return StageResult.escalate(
                context=enriched_context,
                raw_text=user_input.text,
            )

        if not resolved_ids:
            _LOGGER.debug("[Stage0] No concrete targets resolved → escalate.")
            return StageResult.escalate(
                context=enriched_context,
                raw_text=user_input.text,
            )

        # 5. Single entity match → success (ready for execution)
        if len(resolved_ids) == 1:
            _LOGGER.debug(
                "[Stage0] Single target resolved: %s → success",
                resolved_ids[0]
            )
            return StageResult.success(
                intent=intent_name,
                entity_ids=resolved_ids,
                params=self._extract_params(norm_entities),
                context=enriched_context,
                raw_text=user_input.text,
            )

        # 6. Multiple entities → success (ExecutionPipeline handles disambiguation)
        _LOGGER.debug(
            "[Stage0] %d candidates → success (disambiguation in ExecutionPipeline)",
            len(resolved_ids)
        )
        return StageResult.success(
            intent=intent_name,
            entity_ids=resolved_ids,
            params=self._extract_params(norm_entities),
            context=enriched_context,
            raw_text=user_input.text,
        )

    # ----------------------------------------------------------------
    # LEGACY: run() method for backward compatibility during migration
    # TODO: Remove after all callers are updated to use process()
    # ----------------------------------------------------------------
    
    async def run(self, user_input: conversation.ConversationInput, prev_result=None):
        """LEGACY: Old interface for backward compatibility.
        
        This method will be removed after migration is complete.
        New code should use process() instead.
        """
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

        # Inject implied constraints based on intent
        implications = self.INTENT_IMPLICATIONS.get(intent_name, {})
        if implications:
            _LOGGER.debug(
                "[Stage0] Injecting implied constraints for %s: %s",
                intent_name,
                implications,
            )
            norm_entities.update(implications)

        if norm_entities:
            _LOGGER.debug(
                "[Stage0] NLU extracted entities (raw) keys=%s",
                list(norm_entities.keys()),
            )

        # Resolve entities
        resolver = EntityResolverCapability(self.hass, self.config)
        resolved = await resolver.run(user_input, entities=norm_entities)
        resolved_ids = (resolved or {}).get("resolved_ids", [])
        _LOGGER.debug(
            "[Stage0] Entity resolver returned %d id(s): %s",
            len(resolved_ids),
            resolved_ids,
        )

        # Prepare Stage0Result once
        result = Stage0Result(
            type=("intent" if resolved_ids else "clarification"),
            intent=intent_name,
            raw=user_input.text,
            resolved_ids=resolved_ids,
        )

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

        if not resolved_ids:
            _LOGGER.debug("[Stage0] No concrete targets resolved → escalate.")
            return {"status": "escalate", "result": result}

        # Direct Execution Path (Single Entity Match)
        if len(resolved_ids) == 1 and intent_name:
            try:
                _LOGGER.debug(
                    "[Stage0] Direct execution path: intent='%s', single target=%s → delegating to IntentExecutorCapability.",
                    intent_name,
                    resolved_ids[0],
                )

                execution_params = self._extract_params(norm_entities)

                executor = IntentExecutorCapability(self.hass, self.config)
                exec_data = await executor.run(
                    user_input,
                    intent_name=intent_name,
                    entity_ids=resolved_ids,
                    params=execution_params,
                    language=user_input.language or "de",
                )

                if exec_data and exec_data.get("result"):
                    _LOGGER.debug(
                        "[Stage0] Intent executed successfully via capability."
                    )
                    return {"status": "handled", "result": exec_data["result"]}

                _LOGGER.warning(
                    "[Stage0] Intent execution returned no result → escalate."
                )
                return {"status": "escalate", "result": result}

            except Exception as err:
                _LOGGER.exception("[Stage0] Intent execution crashed: %s", err)
                return {
                    "status": "error",
                    "result": await error_response(
                        user_input, f"Interner Fehler beim Ausführen: {err}"
                    ),
                }

        _LOGGER.debug(
            "[Stage0] %d candidate(s) and intent='%s' → escalate to Stage1 for disambiguation.",
            len(resolved_ids),
            intent_name,
        )
        return {"status": "escalate", "result": result}
