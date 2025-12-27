"""Command processor for orchestrating intent execution.

Handles the flow: Filter -> Plural Check -> Disambiguation -> Execution -> Confirmation
"""

import logging
from typing import Any, Dict, List
from .base import Capability
from .intent_executor import IntentExecutorCapability
from .intent_confirmation import IntentConfirmationCapability
from .disambiguation import DisambiguationCapability
from .plural_detection import PluralDetectionCapability
from .disambiguation_select import DisambiguationSelectCapability
from .memory import MemoryCapability

from custom_components.multistage_assist.conversation_utils import (
    make_response,
    error_response,
    filter_candidates_by_state,
)

_LOGGER = logging.getLogger(__name__)

# ============================================================================
# INTENTS THAT SHOULD NEVER BE CACHED
# ============================================================================
# Timer and calendar commands have variable context (names, descriptions)
# that can't be generalized. See stage1_cache.py for detailed explanation.
# ============================================================================
NOCACHE_INTENTS = {"HassTimerSet", "HassTimerCancel"}


class CommandProcessorCapability(Capability):
    """
    Orchestrates the execution pipeline:
    Filter -> Plural Check -> Disambiguation -> Execution -> Confirmation
    
    Called by ExecutionPipeline for each resolved intent+entities.
    """

    name = "command_processor"

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.executor = IntentExecutorCapability(hass, config)
        self.confirmation = IntentConfirmationCapability(hass, config)
        self.disambiguation = DisambiguationCapability(hass, config)
        self.plural = PluralDetectionCapability(hass, config)
        self.select = DisambiguationSelectCapability(hass, config)
        self.memory = MemoryCapability(hass, config)
        self.semantic_cache = None  # Injected by ExecutionPipeline
    
    def set_cache(self, cache):
        """Inject semantic cache capability for storing verified commands."""
        self.semantic_cache = cache

    async def process(
        self,
        user_input,
        candidates: List[str],
        intent_name: str,
        params: Dict[str, Any],
        learning_data=None,
        from_cache: bool = False,  # Skip storing if this came from cache lookup
    ) -> Dict[str, Any]:
        """Main entry point to process a command with candidate entities.
        
        Returns:
            Dict with 'status', 'result', and optionally 'pending_data' for multi-turn flows
        """
        # 1. Filter by State FIRST (before plural detection)
        # This ensures "alle Lichter aus" only targets lights that are ON
        filtered = filter_candidates_by_state(self.hass, candidates, intent_name)
        final_candidates = filtered if filtered else candidates

        # 2. Single Candidate - execute directly
        if len(final_candidates) == 1:
            return await self._execute_final(
                user_input, final_candidates, intent_name, params, learning_data,
                from_cache=from_cache,
            )

        # 3. Plural Detection (on filtered candidates)
        pd = await self.plural.run(user_input) or {}
        if pd.get("multiple_entities") is True:
            return await self._execute_final(
                user_input, final_candidates, intent_name, params, learning_data,
                from_cache=from_cache,
            )

        # 4. Disambiguation needed
        entities_map = {
            eid: self.hass.states.get(eid).attributes.get("friendly_name", eid)
            for eid in final_candidates
        }
        msg_data = await self.disambiguation.run(user_input, entities=entities_map)

        # Return pending state for multi-turn handling
        prompt_message = msg_data.get("message", QUESTION_TEMPLATES["entity"])
        return {
            "status": "handled",
            "result": await make_response(prompt_message, user_input),
            "pending_data": {
                "type": "disambiguation",
                "original_prompt": prompt_message,  # Store for re-prompt
                "candidates": entities_map,
                "intent": intent_name,
                "params": params,
                "learning_data": learning_data,
            },
        }

    async def continue_disambiguation(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle the user's selection from disambiguation."""
        candidates = [
            {"entity_id": eid, "name": name, "ordinal": i + 1}
            for i, (eid, name) in enumerate(pending_data["candidates"].items())
        ]

        selected = await self.select.run(user_input, candidates=candidates)
        if not selected:
            # Re-prompt with same question instead of returning error
            # This keeps the conversation in disambiguation mode
            _LOGGER.debug("[CommandProcessor] Empty selection, re-prompting disambiguation")
            msg_data = await self.disambiguation.run(user_input, entities=pending_data["candidates"])
            return {
                "status": "handled",
                "result": await make_response(
                    msg_data.get("message", SYSTEM_MESSAGES["which_device"]), user_input
                ),
                "pending_data": pending_data,  # Keep the same pending data!
            }

        return await self._execute_final(
            user_input,
            selected,
            pending_data.get("intent", ""),
            pending_data.get("params", {}),
            pending_data.get("learning_data"),
            is_disambiguation_response=True,  # Mark as disambiguation follow-up
        )

    async def re_prompt_pending(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Re-ask pending question after timeout."""
        original_prompt = pending_data.get("original_prompt", SYSTEM_MESSAGES["did_not_understand"])
        message = f"{SYSTEM_MESSAGES['did_not_understand']} {original_prompt}"
        
        return {
            "status": "handled",
            "result": await make_response(message, user_input),
            "pending_data": pending_data,
        }


    async def _execute_final(
        self, user_input, entity_ids, intent_name, params, learning_data=None,
        is_disambiguation_response: bool = False,
        from_cache: bool = False,
    ):
        """Execute intent on entities and generate confirmation."""
        exec_data = await self.executor.run(
            user_input, intent_name=intent_name, entity_ids=entity_ids, params=params
        )

        _LOGGER.debug(
            "[CommandProcessor] IntentExecutor returned: %s",
            {k: v for k, v in (exec_data or {}).items() if k != "result"}
        )

        if not exec_data or "result" not in exec_data:
            _LOGGER.error(
                "[CommandProcessor] IntentExecutor failed - exec_data=%s",
                exec_data
            )
            return {
                "status": "error",
                "result": await error_response(user_input, SYSTEM_MESSAGES["error_short"]),
            }

        result_obj = exec_data["result"]

        # USE EXECUTED PARAMS if available (contains actual brightness values)
        final_params = exec_data.get("executed_params", params)
        
        # Check for verification failures - show error instead of success confirmation
        verification_failures = exec_data.get("verification_failures", [])
        if verification_failures:
            # Generate error message for failed devices
            from ..conversation_utils import join_names
            failed_names = []
            for eid in verification_failures:
                state = self.hass.states.get(eid)
                if state:
                    failed_names.append(state.attributes.get("friendly_name", eid.split(".")[-1]))
                else:
                    failed_names.append(eid.split(".")[-1])
            
            names_str = join_names(failed_names)
            error_msg = f"{names_str} reagiert nicht."
            result_obj.response.async_set_speech(error_msg)
            _LOGGER.warning("[CommandProcessor] Verification failed for: %s", failed_names)
        else:
            # Confirmation only if verification succeeded
            speech = (
                result_obj.response.speech.get("plain", {}).get("speech", "")
                if result_obj.response.speech
                else ""
            )
            if not speech or speech.strip() == "Okay.":
                # Pass final_params instead of original params
                conf_data = await self.confirmation.run(
                    user_input,
                    intent_name=intent_name,
                    entity_ids=entity_ids,
                    params=final_params,
                )
                if conf_data.get("message"):
                    result_obj.response.async_set_speech(conf_data["message"])

        # Multi-turn: Learning Question
        pending_data = None
        if learning_data:
            original = result_obj.response.speech.get("plain", {}).get("speech", "")
            src, tgt = learning_data["source"], learning_data["target"]
            if learning_data.get("type") == "entity":
                new_speech = f"{original} {SYSTEM_MESSAGES['learning_offer_entity'].format(src=src, tgt=tgt)}"
            else:
                new_speech = f"{original} {SYSTEM_MESSAGES['learning_offer_area'].format(src=src, tgt=tgt)}"
            result_obj.response.async_set_speech(new_speech)
            result_obj.continue_conversation = True
            pending_data = {
                "type": "learning_confirmation",
                "learning_type": learning_data.get("type", "area"),
                "source": src,
                "target": tgt,
            }

        # --- Semantic Cache Storage ---
        # Only cache if:
        # 1. Execution was verified successful (no error flag, no verification failures)
        # 2. Command did NOT come from cache (avoid re-caching potentially wrong entries)
        # 3. Intent is not in NOCACHE_INTENTS (timer/calendar need full LLM handling)
        skip_store = from_cache or intent_name in NOCACHE_INTENTS
        if self.semantic_cache and not exec_data.get("error") and not verification_failures and not skip_store:
            try:
                await self.semantic_cache.store(
                    text=user_input.text,
                    intent=intent_name,
                    entity_ids=entity_ids,
                    slots=final_params,
                    required_disambiguation=False,
                    verified=True,
                    is_disambiguation_response=is_disambiguation_response,
                )

            except Exception as e:
                _LOGGER.warning("[CommandProcessor] Failed to cache command: %s", e)

        res = {"status": "handled", "result": result_obj}
        if pending_data:
            res["pending_data"] = pending_data
        return res
