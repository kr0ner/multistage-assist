import logging
from typing import Any, Dict, List
from homeassistant.components import conversation
from .base_stage import BaseStage

# Capabilities
from .capabilities.clarification import ClarificationCapability
from .capabilities.disambiguation import DisambiguationCapability
from .capabilities.disambiguation_select import DisambiguationSelectCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .capabilities.intent_confirmation import IntentConfirmationCapability  # <--- Keep
from .capabilities.intent_executor import IntentExecutorCapability
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.keyword_intent import KeywordIntentCapability
from .capabilities.area_alias import AreaAliasCapability
# from .capabilities.response_generator import ResponseGeneratorCapability <--- REMOVE
from .capabilities.memory import MemoryCapability
from .capabilities.intent_resolution import IntentResolutionCapability

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
        IntentConfirmationCapability, # <--- Keep
        IntentExecutorCapability,
        # ResponseGeneratorCapability, <--- REMOVE
        MemoryCapability,
        IntentResolutionCapability,
        EntityResolverCapability,
        KeywordIntentCapability,
        AreaAliasCapability
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    async def _add_confirmation_if_needed(self, user_input, result, intent_name, entity_ids, params=None):
        """Helper to inject natural speech if missing."""
        if not result or not result.response:
            return

        speech = result.response.speech.get("plain", {}).get("speech", "") if result.response.speech else ""
        
        # If execution returned empty or generic "Okay", generate a nice sentence
        if not speech or speech.strip() == "Okay.":
            confirm_cap = self.get("intent_confirmation")
            # Pass params so it knows "50%" or "red"
            gen_data = await confirm_cap.run(user_input, intent_name=intent_name, entity_ids=entity_ids, params=params)
            msg = gen_data.get("message")
            if msg:
                result.response.async_set_speech(msg)

    def _merge_speech(self, target_result, source_results):
        texts = []
        for r in source_results:
            resp = getattr(r, "response", None)
            if resp:
                s = getattr(resp, "speech", {})
                plain = s.get("plain", {}).get("speech", "")
                if plain:
                    texts.append(plain)
        
        target_resp = getattr(target_result, "response", None)
        if target_resp:
            s = getattr(target_resp, "speech", {})
            target_text = s.get("plain", {}).get("speech", "")
            if target_text:
                texts.append(target_text)
            
            full_text = " ".join(texts)
            if full_text:
                target_resp.async_set_speech(full_text)

    async def _execute_sequence(self, user_input, commands: List[str], previous_results: List[Any] = None) -> Dict[str, Any]:
        results = list(previous_results) if previous_results else []
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        agent = getattr(self, "agent", None)

        for i, cmd in enumerate(commands):
            _LOGGER.debug("[Stage1] Executing atomic command %d/%d: %s", i + 1, len(commands), cmd)
            sub_input = with_new_text(user_input, cmd)
            
            result = await agent._run_pipeline(sub_input)
            
            if key in self._pending:
                remaining = commands[i+1:]
                if remaining:
                    self._pending[key]["remaining"] = remaining
                
                if results:
                    self._merge_speech(result, results)
                return {"status": "handled", "result": result}
            
            results.append(result)

        if not results:
             return {"status": "error", "result": await error_response(user_input, "Keine Befehle ausgeführt.")}
        
        final = results[-1]
        self._merge_speech(final, results[:-1])
        return {"status": "handled", "result": final}

    async def _process_candidates(self, user_input, candidates, intent_name, params, learning_data=None):
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        
        if len(candidates) == 1:
            return await self._execute_final(user_input, candidates, intent_name, params, learning_data=learning_data)

        pd = await self.use("plural_detection", user_input) or {}
        if pd.get("multiple_entities") is True:
             return await self._execute_final(user_input, candidates, intent_name, params, learning_data=learning_data)

        filtered = filter_candidates_by_state(self.hass, candidates, intent_name)
        final = filtered if filtered else candidates
        
        if len(final) == 1:
            return await self._execute_final(user_input, final, intent_name, params, learning_data=learning_data)

        # Disambiguation
        entities_map = {eid: self.hass.states.get(eid).attributes.get("friendly_name", eid) for eid in final}
        msg_data = await self.use("disambiguation", user_input, entities=entities_map)
        self._pending[key] = {"candidates": entities_map, "intent": intent_name, "params": params, "raw": user_input.text}
        return {"status": "handled", "result": await make_response(msg_data.get("message", "Welches Gerät?"), user_input)}

    async def _execute_final(self, user_input, entity_ids, intent_name, params, remaining=None, learning_data=None):
        exec_data = await self.use("intent_executor", user_input, intent_name=intent_name, entity_ids=entity_ids, params=params, language=user_input.language or "de")
        
        if not exec_data or "result" not in exec_data:
            return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        result_obj = exec_data["result"]

        # A. Inject Natural Confirmation
        await self._add_confirmation_if_needed(user_input, result_obj, intent_name, entity_ids, params)

        # B. Inject Learning Question
        if learning_data:
             key = getattr(user_input, "session_id", None) or user_input.conversation_id
             original_speech = result_obj.response.speech.get("plain", {}).get("speech", "")
             src, tgt = learning_data["source"], learning_data["target"]
             new_speech = f"{original_speech} Übrigens, ich habe '{src}' als '{tgt}' interpretiert. Soll ich mir das merken?"
             result_obj.response.async_set_speech(new_speech)
             result_obj.continue_conversation = True
             self._pending[key] = {"type": "learning_confirmation", "source": src, "target": tgt}

        if remaining:
             return await self._execute_sequence(user_input, remaining, previous_results=[result_obj])
        
        return {"status": "handled", "result": result_obj}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s', prev_result=%s", user_input.text, type(prev_result).__name__)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # 1. Handle Pending (Disambiguation / Learning)
        if key in self._pending:
            return await self._handle_pending(key, user_input)

        # 2. Handle Stage0 Handoff (Refining NLU results)
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            return await self._handle_stage0_result(prev_result, user_input)

        # 3. Handle New Command
        return await self._handle_new_command(user_input, prev_result)

    # --- Sub-Handlers ---

    async def _handle_pending(self, key: str, user_input) -> Dict[str, Any]:
        pending = self._pending.pop(key, None)
        if not pending: return {"status": "error", "result": await error_response(user_input)}

        # Learning Confirmation
        if pending.get("type") == "learning_confirmation":
            if user_input.text.lower().strip() in ("ja", "ja bitte", "gerne", "okay", "mach das"):
                await self.get("memory").learn_area_alias(pending["source"], pending["target"])
                return {"status": "handled", "result": await make_response("Alles klar, gemerkt.", user_input)}
            return {"status": "handled", "result": await make_response("Okay, nicht gemerkt.", user_input)}

        # Disambiguation
        candidates = [{"entity_id": eid, "name": name, "ordinal": i + 1} 
                      for i, (eid, name) in enumerate(pending["candidates"].items())]
        selected = await self.use("disambiguation_select", user_input, candidates=candidates)
        
        if not selected:
            return {"status": "error", "result": await error_response(user_input)}

        return await self._execute_final(
            user_input, 
            selected, 
            pending.get("intent", ""), 
            pending.get("params", {}),
            pending.get("remaining")
        )

    async def _handle_stage0_result(self, prev_result: Stage0Result, user_input) -> Dict[str, Any]:
        """Refine Stage0 results using IntentResolution."""
        res_data = await self.use("intent_resolution", user_input)
        
        if res_data:
            return await self._process_candidates(
                user_input, 
                res_data["entity_ids"], 
                res_data["intent"], 
                {k: v for k, v in res_data["slots"].items() if k not in ("name", "entity_id")},
                res_data.get("learning_data")
            )
        
        return await self._process_candidates(
            user_input, 
            list(prev_result.resolved_ids), 
            (prev_result.intent or "").strip(),
            {}
        )

    async def _handle_new_command(self, user_input, prev_result) -> Dict[str, Any]:
        """Clarify and Execute."""
        clar_data = await self.use("clarification", user_input)
        
        # If Empty -> Chat
        if not clar_data or (isinstance(clar_data, list) and not clar_data):
            return {"status": "escalate", "result": prev_result}

        if isinstance(clar_data, list):
            norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clar_data if isinstance(c, str) and c.strip()]

            # 1. Exact Match Single Command -> Resolve & Execute
            if len(atomic) == 1 and atomic[0].strip().lower() == norm:
                if isinstance(prev_result, Stage0Result) and prev_result.intent:
                     return {"status": "escalate", "result": prev_result}

                res_data = await self.use("intent_resolution", user_input)
                
                if not res_data:
                    return {"status": "escalate", "result": prev_result}

                params = {k: v for (k, v) in res_data["slots"].items() if k not in ("name", "entity_id")}
                return await self._process_candidates(
                    user_input, 
                    res_data["entity_ids"], 
                    res_data["intent"], 
                    params, 
                    res_data.get("learning_data")
                )

            # 2. Multi Command OR Rephrased Single Command
            if len(atomic) > 0:
                return await self._execute_sequence(user_input, atomic)

        return {"status": "escalate", "result": prev_result}