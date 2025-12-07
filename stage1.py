import logging
from typing import Any, Dict, List
from homeassistant.components import conversation
from .base_stage import BaseStage

# Capabilities
from .capabilities.clarification import ClarificationCapability
from .capabilities.disambiguation import DisambiguationCapability
from .capabilities.disambiguation_select import DisambiguationSelectCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .capabilities.intent_confirmation import IntentConfirmationCapability
from .capabilities.intent_executor import IntentExecutorCapability
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.keyword_intent import KeywordIntentCapability
from .capabilities.area_alias import AreaAliasCapability
from .capabilities.memory import MemoryCapability
from .capabilities.intent_resolution import IntentResolutionCapability
from .capabilities.timer import TimerCapability
from .capabilities.command_processor import CommandProcessorCapability

from .conversation_utils import make_response, error_response, with_new_text, filter_candidates_by_state
from .stage_result import Stage0Result

_LOGGER = logging.getLogger(__name__)


class Stage1Processor(BaseStage):
    """Stage 1: Handles clarification, disambiguation, and multi-command orchestration."""

    name = "stage1"
    capabilities = [
        ClarificationCapability,
        DisambiguationCapability,
        DisambiguationSelectCapability,
        PluralDetectionCapability,
        IntentConfirmationCapability,
        IntentExecutorCapability,
        EntityResolverCapability,
        KeywordIntentCapability,
        AreaAliasCapability,
        MemoryCapability,
        IntentResolutionCapability,
        TimerCapability,
        CommandProcessorCapability
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s'", user_input.text)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # 1. Handle Pending
        if key in self._pending:
            return await self._handle_pending(key, user_input)

        # 2. Handle Stage0
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            return await self._handle_stage0_result(prev_result, user_input)

        # 3. Handle New Command
        return await self._handle_new_command(user_input, prev_result)

    # --- Handlers ---

    async def _handle_pending(self, key: str, user_input) -> Dict[str, Any]:
        pending = self._pending.pop(key, None)
        if not pending: return {"status": "error", "result": await error_response(user_input)}

        ptype = pending.get("type")

        # Timer Follow-up
        if ptype == "timer":
            timer = self.get("timer")
            res = await timer.continue_flow(user_input, pending)
            
            # If flow not done, keep pending
            if res.get("pending_data"):
                self._pending[key] = res["pending_data"]

            # --- NEW: Check for Learning Data from Timer ---
            if res.get("status") == "handled" and res.get("learning_data"):
                l_data = res["learning_data"]
                original_speech = res["result"].response.speech.get("plain", {}).get("speech", "")
                
                # Format friendly message
                src, tgt = l_data["source"], l_data["target"]
                # Clean up target name (e.g. notify.mobile_app_sm_a546b -> SM A546B)
                tgt_display = tgt.replace("notify.mobile_app_", "").replace("_", " ").title()

                new_speech = f"{original_speech} Übrigens, ich habe '{src}' als '{tgt_display}' interpretiert. Soll ich mir das merken?"
                res["result"].response.async_set_speech(new_speech)
                res["result"].continue_conversation = True
                
                # Set pending state for confirmation
                self._pending[key] = {
                    "type": "learning_confirmation", 
                    "learning_type": "entity", # Treat devices as entities
                    "source": src, 
                    "target": tgt
                }
            # -----------------------------------------------

            return self._handle_processor_result(key, res)

        # Learning Confirmation
        if ptype == "learning_confirmation":
            if user_input.text.lower().strip() in ("ja", "ja bitte", "gerne", "okay", "mach das"):
                memory = self.get("memory")
                if pending.get("learning_type") == "entity":
                    await memory.learn_entity_alias(pending["source"], pending["target"])
                else:
                    await memory.learn_area_alias(pending["source"], pending["target"])
                return {"status": "handled", "result": await make_response("Alles klar, gemerkt.", user_input)}
            return {"status": "handled", "result": await make_response("Okay, nicht gemerkt.", user_input)}

        # Disambiguation
        if ptype == "disambiguation":
            processor = self.get("command_processor")
            res = await processor.continue_disambiguation(user_input, pending)
            if res.get("status") == "handled" and pending.get("remaining"):
                 return await self._execute_sequence(user_input, pending["remaining"], previous_results=[res["result"]])
            return self._handle_processor_result(key, res)

        return {"status": "error", "result": await error_response(user_input)}

    async def _handle_new_command(self, user_input, prev_result) -> Dict[str, Any]:
        clar_data = await self.use("clarification", user_input)
        
        if not clar_data or (isinstance(clar_data, list) and not clar_data):
            return {"status": "escalate", "result": prev_result}

        if isinstance(clar_data, list):
            norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clar_data if isinstance(c, str) and c.strip()]

            if len(atomic) == 1 and atomic[0].strip().lower() == norm:
                if isinstance(prev_result, Stage0Result) and prev_result.intent:
                     return {"status": "escalate", "result": prev_result}

                ki_data = await self.use("keyword_intent", user_input) or {}
                intent_name = ki_data.get("intent")
                slots = ki_data.get("slots") or {}

                # --- TIMER INTERCEPT ---
                if intent_name == "HassTimerSet":
                     target_name = slots.get("name")
                     # Check memory for device name
                     if target_name:
                         memory = self.get("memory")
                         known_id = await memory.get_entity_alias(target_name)
                         if known_id:
                             _LOGGER.debug("[Stage1] Memory hit for timer device: %s -> %s", target_name, known_id)
                             slots["device_id"] = known_id

                     res = await self.get("timer").run(user_input, intent_name, slots)
                     
                     # Note: run() doesn't usually return learning_data directly unless it did fuzzy match immediately?
                     # TimerCapability currently does fuzzy match in _process_request but doesn't return learning_data there yet.
                     # We should update TimerCapability to return learning_data in _process_request too if fuzzy matched!
                     
                     # But for now, handle pending/result
                     return self._handle_processor_result(
                        getattr(user_input, "session_id", None) or user_input.conversation_id,
                        res
                    )
                # -----------------------

                res_data = await self.use("intent_resolution", user_input)
                if not res_data:
                    return {"status": "escalate", "result": prev_result}

                processor = self.get("command_processor")
                res = await processor.process(
                    user_input, 
                    res_data["entity_ids"], 
                    res_data["intent"], 
                    {k: v for k, v in res_data["slots"].items() if k not in ("name", "entity_id")},
                    res_data.get("learning_data")
                )
                return self._handle_processor_result(getattr(user_input, "session_id", None) or user_input.conversation_id, res)

            if len(atomic) > 0:
                return await self._execute_sequence(user_input, atomic)

        return {"status": "escalate", "result": prev_result}

    def _handle_processor_result(self, key, res: Dict[str, Any]) -> Dict[str, Any]:
        if res.get("pending_data"):
            self._pending[key] = res["pending_data"]
        return res

    async def _execute_sequence(self, user_input, commands: List[str], previous_results: List[Any] = None) -> Dict[str, Any]:
        results = list(previous_results) if previous_results else []
        agent = getattr(self, "agent", None)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        for i, cmd in enumerate(commands):
            _LOGGER.debug("[Stage1] Sequence %d/%d: %s", i+1, len(commands), cmd)
            res = await agent._run_pipeline(with_new_text(user_input, cmd))
            if key in self._pending:
                remaining = commands[i+1:]
                self._pending[key]["remaining"] = remaining
                if results: self._merge_speech(res, results)
                return {"status": "handled", "result": res}
            results.append(res)
        
        final = results[-1]
        self._merge_speech(final, results[:-1])
        return {"status": "handled", "result": final}

    def _merge_speech(self, target_result, source_results):
        texts = []
        for r in source_results:
            if not r: continue
            resp = getattr(r, "response", None)
            if resp:
                s = getattr(resp, "speech", {})
                plain = s.get("plain", {}).get("speech", "")
                if plain: texts.append(plain)
        target_resp = getattr(target_result, "response", None)
        if target_resp:
            s = getattr(target_resp, "speech", {})
            target_text = s.get("plain", {}).get("speech", "")
            if target_text: texts.append(target_text)
            full_text = " ".join(texts)
            if full_text: target_resp.async_set_speech(full_text)

    # _filter_candidates_by_state and _add_confirmation_if_needed are no longer needed here 
    # as they are handled in CommandProcessor, but _execute_sequence uses _merge_speech locally.
    # _add_confirmation_if_needed logic in Stage1 was redundant for the main flow
    # but required if we called it directly. Since we use CommandProcessor now for standard flow,
    # we can remove _add_confirmation_if_needed from Stage1 class to clean up.
    # Wait, _handle_pending -> Disambiguation calls it? No, Disambig uses CommandProcessor.
    # So we can remove _add_confirmation_if_needed from Stage1!
    
    # Keeping _handle_stage0_result for completeness
    async def _handle_stage0_result(self, prev_result: Stage0Result, user_input) -> Dict[str, Any]:
        res_data = await self.use("intent_resolution", user_input)
        candidates = list(prev_result.resolved_ids)
        intent_name = (prev_result.intent or "").strip()
        params = {}

        if res_data:
             candidates = res_data["entity_ids"]
             intent_name = res_data["intent"]
             params = {k: v for k, v in res_data["slots"].items() if k not in ("name", "entity_id")}

        processor = self.get("command_processor")
        res = await processor.process(user_input, candidates, intent_name, params, res_data.get("learning_data") if res_data else None)
        return self._handle_processor_result(getattr(user_input, "session_id", None) or user_input.conversation_id, res)