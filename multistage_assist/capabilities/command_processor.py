import logging
from typing import Any, Dict, List
from .base import Capability
from .intent_executor import IntentExecutorCapability
from .intent_confirmation import IntentConfirmationCapability
from .disambiguation import DisambiguationCapability
from .plural_detection import PluralDetectionCapability
from .disambiguation_select import DisambiguationSelectCapability
from .memory import MemoryCapability
from .timer import TimerCapability

from custom_components.multistage_assist.conversation_utils import (
    make_response,
    error_response,
    with_new_text,
    filter_candidates_by_state,
)

_LOGGER = logging.getLogger(__name__)


class CommandProcessorCapability(Capability):
    """
    Orchestrates the execution pipeline:
    Filter -> Plural Check -> Disambiguation -> Execution -> Confirmation
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
        self.timer = TimerCapability(hass, config)
        self.semantic_cache = None  # Injected by Stage1
    
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
        agent=None,
        from_cache: bool = False,  # Skip storing if this came from cache lookup
    ) -> Dict[str, Any]:
        """Main entry point to process a command with candidate entities."""

        # 1. Timer Intercept (already handled in Stage1 mostly, but good to have safety)
        if intent_name == "HassTimerSet":
            # Timer handles its own execution flow
            return await self.timer.run(user_input, intent_name, params)

        # 2. Single Candidate Optimization
        if len(candidates) == 1:
            return await self._execute_final(
                user_input, candidates, intent_name, params, learning_data,
                from_cache=from_cache,
            )

        # 3. Filter by State FIRST (before plural detection)
        # This ensures "alle Lichter aus" only targets lights that are ON
        filtered = filter_candidates_by_state(self.hass, candidates, intent_name)
        final_candidates = filtered if filtered else candidates

        # Check single after filtering
        if len(final_candidates) == 1:
            return await self._execute_final(
                user_input, final_candidates, intent_name, params, learning_data,
                from_cache=from_cache,
            )

        # 4. Plural Detection (on filtered candidates)
        pd = await self.plural.run(user_input) or {}
        if pd.get("multiple_entities") is True:
            return await self._execute_final(
                user_input, final_candidates, intent_name, params, learning_data,
                from_cache=from_cache,
            )

        # 5. Disambiguation
        entities_map = {
            eid: self.hass.states.get(eid).attributes.get("friendly_name", eid)
            for eid in final_candidates
        }
        msg_data = await self.disambiguation.run(user_input, entities=entities_map)

        # Return pending state for Stage1 to store
        return {
            "status": "handled",
            "result": await make_response(
                msg_data.get("message", "Welches Gerät?"), user_input
            ),
            "pending_data": {
                "type": "disambiguation",
                "candidates": entities_map,
                "intent": intent_name,
                "params": params,
                "learning_data": learning_data,
            },
        }

    async def continue_disambiguation(
        self, user_input, pending_data: Dict[str, Any], agent=None
    ) -> Dict[str, Any]:
        """Handle the user's selection from disambiguation."""
        candidates = [
            {"entity_id": eid, "name": name, "ordinal": i + 1}
            for i, (eid, name) in enumerate(pending_data["candidates"].items())
        ]

        selected = await self.select.run(user_input, candidates=candidates)
        if not selected:
            return {"status": "error", "result": await error_response(user_input)}

        return await self._execute_final(
            user_input,
            selected,
            pending_data.get("intent", ""),
            pending_data.get("params", {}),
            pending_data.get("learning_data"),
            is_disambiguation_response=True,  # Mark as disambiguation follow-up
        )

    async def _execute_final(
        self, user_input, entity_ids, intent_name, params, learning_data=None,
        is_disambiguation_response: bool = False,
        from_cache: bool = False,
    ):
        exec_data = await self.executor.run(
            user_input, intent_name=intent_name, entity_ids=entity_ids, params=params
        )

        if not exec_data or "result" not in exec_data:
            return {
                "status": "error",
                "result": await error_response(user_input, "Fehler."),
            }

        result_obj = exec_data["result"]

        # USE EXECUTED PARAMS if available (contains actual brightness values)
        final_params = exec_data.get("executed_params", params)

        # Confirmation
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

        # Inject Learning Question (Same as before)
        pending_data = None
        if learning_data:
            original = result_obj.response.speech.get("plain", {}).get("speech", "")
            src, tgt = learning_data["source"], learning_data["target"]
            t_type = "Gerät" if learning_data.get("type") == "entity" else "Bereich"
            new_speech = f"{original} Übrigens, ich habe '{src}' als {t_type} '{tgt}' interpretiert. Soll ich mir das merken?"
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
        # 1. Execution was verified successful (no error flag)
        # 2. Command did NOT come from cache (avoid re-caching potentially wrong entries)
        if self.semantic_cache and not exec_data.get("error") and not from_cache:
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

    async def execute_sequence(
        self, user_input, commands: List[str], agent
    ) -> Dict[str, Any]:
        """Execute a list of atomic commands."""
        results = []
        for i, cmd in enumerate(commands):
            _LOGGER.debug(
                "[CommandProcessor] Sequence %d/%d: %s", i + 1, len(commands), cmd
            )
            # We call back into the agent to run the full pipeline for each sub-command
            # This allows full resolution/clarification for each part
            res = await agent._run_pipeline(with_new_text(user_input, cmd))

            # If any step returns pending/error, we stop?
            # Actually, if pending, we must stop and return the pending state to the user
            # But we need to preserve the "rest of the sequence"

            # This logic is tricky to move completely out of Stage1 without passing 'agent'
            # or having a recursive dependency.
            # Passed 'agent' in signature.

            # Check if result indicates pending?
            # The result from _run_pipeline is a ConversationResult or None.
            # It DOES NOT return the internal "status/pending" dict.
            # Stage1 wraps it. This makes extracting sequence logic hard.

            # Better to keep sequence logic in Stage1 for now, or change _run_pipeline return signature.
            results.append(res)

        # Merge speech manually? The original logic merged speech.
        # Let's keep sequence in Stage1 for simplicity for now.
        return {}
