import logging
from typing import Any, Dict, List, Optional
from homeassistant.components import conversation
from homeassistant.helpers import intent as ha_intent
from .base_stage import BaseStage
from .capabilities.clarification import ClarificationCapability
from .capabilities.disambiguation import DisambiguationCapability
from .capabilities.disambiguation_select import DisambiguationSelectCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .capabilities.intent_confirmation import IntentConfirmationCapability
from .capabilities.intent_executor import IntentExecutorCapability
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.keyword_intent import KeywordIntentCapability
from .capabilities.area_alias import AreaAliasCapability
from .capabilities.response_generator import ResponseGeneratorCapability
from .capabilities.chat import ChatCapability
from .conversation_utils import make_response, error_response, with_new_text
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
        ResponseGeneratorCapability,
        ChatCapability
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _filter_candidates_by_state(self, entity_ids: List[str], intent_name: str) -> List[str]:
        """Filter out entities that are already in the desired state."""
        if intent_name not in ("HassTurnOn", "HassTurnOff"):
            return entity_ids

        filtered = []
        for eid in entity_ids:
            state_obj = self.hass.states.get(eid)
            if not state_obj or state_obj.state in ("unavailable", "unknown"):
                continue

            state = state_obj.state
            domain = eid.split(".", 1)[0]
            
            is_relevant = False
            
            if intent_name == "HassTurnOff":
                if domain == "cover":
                    if state != "closed": is_relevant = True
                elif state != "off": 
                    is_relevant = True

            elif intent_name == "HassTurnOn":
                if domain == "cover":
                    if state != "open": is_relevant = True
                elif state != "on":
                    is_relevant = True
            
            if is_relevant:
                filtered.append(eid)
        
        return filtered

    async def _add_confirmation_if_needed(self, user_input, result, intent_name, entity_ids):
        """Helper to inject random speech if missing."""
        if not result or not result.response:
            return

        speech = result.response.speech.get("plain", {}).get("speech", "") if result.response.speech else ""
        
        if not speech or speech.strip() == "Okay":
            resp_gen = self.get("response_generator")
            gen_data = await resp_gen.run(user_input, intent_name=intent_name, entity_ids=entity_ids)
            msg = gen_data.get("message")
            if msg:
                result.response.async_set_speech(msg)

    def _merge_speech(self, target_result, source_results):
        """Prepend speech from source_results to target_result."""
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
        """Execute a list of atomic commands sequentially."""
        results = list(previous_results) if previous_results else []
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        agent = getattr(self, "agent", None)

        for i, cmd in enumerate(commands):
            _LOGGER.debug("[Stage1] Executing atomic command %d/%d: %s", i + 1, len(commands), cmd)
            sub_input = with_new_text(user_input, cmd)
            
            result = await agent._run_pipeline(sub_input)
            
            if key in self._pending:
                _LOGGER.debug("[Stage1] Command '%s' triggered pending state.", cmd)
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

    async def _process_candidates(self, user_input, candidates: List[str], intent_name: str, params: Dict[str, Any] = None):
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        
        if len(candidates) == 1:
            try:
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=intent_name,
                    entity_ids=candidates,
                    params=params,
                    language=user_input.language or "de",
                )
                if not exec_data or "result" not in exec_data:
                    return {"status": "error", "result": await error_response(user_input, "Fehler.")}
                
                await self._add_confirmation_if_needed(user_input, exec_data["result"], intent_name, candidates)
                return {"status": "handled", "result": exec_data["result"]}
            except Exception as e:
                _LOGGER.exception("[Stage1] execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        pd = await self.use("plural_detection", user_input) or {}
        if pd.get("multiple_entities") is True:
            try:
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=intent_name,
                    entity_ids=candidates,
                    params=params,
                    language=user_input.language or "de",
                )
                if not exec_data or "result" not in exec_data:
                    return {"status": "error", "result": await error_response(user_input, "Fehler.")}
                
                await self._add_confirmation_if_needed(user_input, exec_data["result"], intent_name, candidates)
                return {"status": "handled", "result": exec_data["result"]}
            except Exception as e:
                _LOGGER.exception("[Stage1] Direct multi-target execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        filtered_ids = self._filter_candidates_by_state(candidates, intent_name)
        final_candidates = filtered_ids if filtered_ids else candidates
        
        if len(final_candidates) == 1:
            try:
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=intent_name,
                    entity_ids=final_candidates,
                    params=params,
                    language=user_input.language or "de",
                )
                if not exec_data or "result" not in exec_data:
                        return {"status": "error", "result": await error_response(user_input, "Fehler.")}
                
                await self._add_confirmation_if_needed(user_input, exec_data["result"], intent_name, final_candidates)
                return {"status": "handled", "result": exec_data["result"]}
            except Exception as e:
                _LOGGER.exception("[Stage1] execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        entities_map = {
            eid: self.hass.states.get(eid).attributes.get("friendly_name", eid)
            for eid in final_candidates
        }
        data = await self.use("disambiguation", user_input, entities=entities_map)
        msg = (data or {}).get("message") or "Welches Gerät meinst du?"

        self._pending[key] = {
            "candidates": entities_map, 
            "intent": intent_name, 
            "params": params,
            "raw": user_input.text
        }
        return {"status": "handled", "result": await make_response(msg, user_input)}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s', prev_result=%s", user_input.text, type(prev_result).__name__)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        if key in self._pending:
            pending = self._pending.pop(key, None)
            if not pending:
                return {"status": "error", "result": await error_response(user_input)}

            candidates = [
                {"entity_id": eid, "name": name, "ordinal": i + 1}
                for i, (eid, name) in enumerate(pending["candidates"].items())
            ]

            selected = await self.use("disambiguation_select", user_input, candidates=candidates)
            if not selected:
                return {"status": "error", "result": await error_response(user_input)}

            intent_name = (pending.get("intent") or "").strip()
            params = pending.get("params", {}) 

            try:
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=intent_name,
                    entity_ids=selected,
                    params=params, 
                    language=user_input.language or "de",
                )
                if not exec_data or "result" not in exec_data:
                    return {"status": "error", "result": await error_response(user_input, "Fehler.")}

                await self._add_confirmation_if_needed(user_input, exec_data["result"], intent_name, selected)

                conv_result = exec_data["result"]
                remaining = pending.get("remaining")
                if remaining:
                    return await self._execute_sequence(user_input, remaining, previous_results=[conv_result])

                return {"status": "handled", "result": conv_result}

            except Exception as e:
                _LOGGER.exception("[Stage1] Direct action execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            candidates = list(prev_result.resolved_ids)
            intent_name = (prev_result.intent or "").strip()
            
            ki_data = await self.use("keyword_intent", user_input) or {}
            strict_domain = ki_data.get("domain")
            strict_intent = ki_data.get("intent")
            
            if strict_domain and strict_intent:
                 slots = ki_data.get("slots") or {}
                 er_data = await self.use("entity_resolver", user_input, entities=slots) or {}
                 refined_ids = er_data.get("resolved_ids") or []
                 
                 if refined_ids:
                     candidates = refined_ids
                     intent_name = strict_intent
            
            return await self._process_candidates(user_input, candidates, intent_name)

        clar_data = await self.use("clarification", user_input)

        # --- FIX: Handle Empty Clarification Result (Chat Fallback) ---
        if not clar_data or (isinstance(clar_data, list) and not clar_data):
            _LOGGER.debug("[Stage1] Clarification returned empty list -> falling back to Chat Capability.")
            chat_res = await self.use("chat", user_input)
            if chat_res:
                return {"status": "handled", "result": chat_res}
            return {"status": "escalate", "result": prev_result}
        # -------------------------------------------------------------

        if isinstance(clar_data, list):
            original_norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clar_data if isinstance(c, str) and c.strip()]

            if len(atomic) == 1 and atomic[0].strip().lower() == original_norm:
                if isinstance(prev_result, Stage0Result) and prev_result.intent and prev_result.type == "intent":
                    return {"status": "escalate", "result": prev_result}

                ki_data = await self.use("keyword_intent", user_input) or {}
                intent_name = ki_data.get("intent")
                slots = ki_data.get("slots") or {}

                if not intent_name:
                    _LOGGER.debug("[Stage1] KeywordIntent found no intent -> falling back to Chat Capability.")
                    chat_res = await self.use("chat", user_input)
                    if chat_res:
                        return {"status": "handled", "result": chat_res}
                    return {"status": "escalate", "result": prev_result}

                er_data = await self.use("entity_resolver", user_input, entities=slots) or {}
                entity_ids = er_data.get("resolved_ids") or []

                if not entity_ids:
                    candidate_area = slots.get("area") or slots.get("name")
                    if candidate_area:
                        alias_res = await self.use("area_alias", user_input, search_text=candidate_area)
                        mapped_area = alias_res.get("area")
                        if mapped_area:
                            new_slots = slots.copy()
                            new_slots["area"] = mapped_area
                            if new_slots.get("name") == candidate_area:
                                new_slots.pop("name") 
                            er_data = await self.use("entity_resolver", user_input, entities=new_slots) or {}
                            entity_ids = er_data.get("resolved_ids") or []
                            slots = new_slots 

                if not entity_ids:
                    # Also fallback to chat if we understood intent but found NO entities (e.g. general question)
                    # BUT ONLY if it wasn't a strict command.
                    # Actually, if we found intent but no entities, it might be safer to escalate or say "I don't know that device".
                    # Let's try chat as a Hail Mary.
                    _LOGGER.debug("[Stage1] Intent found but no entities -> trying Chat.")
                    chat_res = await self.use("chat", user_input)
                    if chat_res:
                        return {"status": "handled", "result": chat_res}
                    return {"status": "escalate", "result": prev_result}

                params = {k: v for (k, v) in slots.items() if k not in ("name", "entity_id")}
                return await self._process_candidates(user_input, entity_ids, intent_name, params)

            if len(atomic) > 1 or (len(atomic) == 1 and atomic[0].strip().lower() != original_norm):
                return await self._execute_sequence(user_input, atomic)

        return {"status": "escalate", "result": prev_result}
