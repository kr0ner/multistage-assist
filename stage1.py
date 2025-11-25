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
        AreaAliasCapability
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

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
        """Execute a list of atomic commands sequentially, stopping on disambiguation."""
        results = list(previous_results) if previous_results else []
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        agent = getattr(self, "agent", None)

        if not agent:
             _LOGGER.error("[Stage1] Agent reference missing in Stage1Processor.")
             return {"status": "error", "result": await error_response(user_input)}

        for i, cmd in enumerate(commands):
            _LOGGER.debug("[Stage1] Executing atomic command %d/%d: %s", i + 1, len(commands), cmd)
            sub_input = with_new_text(user_input, cmd)
            
            result = await agent._run_pipeline(sub_input)
            
            if key in self._pending:
                _LOGGER.debug("[Stage1] Command '%s' triggered pending state. Halting sequence.", cmd)
                remaining = commands[i+1:]
                if remaining:
                    self._pending[key]["remaining"] = remaining
                    _LOGGER.debug("[Stage1] Stored %d remaining commands for later.", len(remaining))
                
                if results:
                    self._merge_speech(result, results)
                return {"status": "handled", "result": result}
            
            results.append(result)

        if not results:
             return {"status": "error", "result": await error_response(user_input, "Keine Befehle ausgeführt.")}
        
        final = results[-1]
        self._merge_speech(final, results[:-1])
        _LOGGER.debug("[Stage1] Sequence execution finished. Merged %d results.", len(results))
        return {"status": "handled", "result": final}

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

    async def _process_candidates(self, user_input, candidates: List[str], intent_name: str, params: Dict[str, Any] = None):
        """Unified logic for handling multiple candidates (filter -> plural -> disambiguate)."""
        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        
        # 1) Plural detection (do this first to support "alle lichter")
        pd = await self.use("plural_detection", user_input) or {}
        if pd.get("multiple_entities") is True:
            _LOGGER.debug("[Stage1] Plural confirmed → executing action for all %d entities.", len(candidates))
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
                return {"status": "handled", "result": exec_data["result"]}
            except Exception as e:
                _LOGGER.exception("[Stage1] Direct multi-target execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        # 2) Context-aware filtering
        filtered_ids = self._filter_candidates_by_state(candidates, intent_name)
        final_candidates = filtered_ids if filtered_ids else candidates
        
        if len(filtered_ids) < len(candidates):
            _LOGGER.debug("[Stage1] State filtering reduced candidates from %d to %d", len(candidates), len(filtered_ids))

        # 3) Check if we have exactly one winner after filtering
        if len(final_candidates) == 1:
            _LOGGER.debug("[Stage1] Auto-selecting single candidate: %s", final_candidates[0])
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
                return {"status": "handled", "result": exec_data["result"]}
            except Exception as e:
                _LOGGER.exception("[Stage1] execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        # 4) Otherwise: Disambiguation
        _LOGGER.debug("[Stage1] Ambiguous candidates (%d) → initiating disambiguation.", len(final_candidates))
        entities_map = {
            eid: self.hass.states.get(eid).attributes.get("friendly_name", eid)
            for eid in final_candidates
        }
        data = await self.use("disambiguation", user_input, entities=entities_map)
        msg = (data or {}).get("message") or "Welches Gerät meinst du?"

        self._pending[key] = {
            "candidates": entities_map, 
            "intent": intent_name, 
            "params": params, # Store params so they are passed on resume
            "raw": user_input.text
        }
        _LOGGER.debug("[Stage1] Stored pending disambiguation context for %s", key)
        return {"status": "handled", "result": await make_response(msg, user_input)}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s', prev_result=%s", user_input.text, type(prev_result).__name__)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # --- Handle disambiguation follow-up ---
        if key in self._pending:
            _LOGGER.debug("[Stage1] Resuming pending disambiguation for key=%s", key)
            pending = self._pending.pop(key, None)
            if not pending:
                return {"status": "error", "result": await error_response(user_input)}

            candidates = [
                {"entity_id": eid, "name": name, "ordinal": i + 1}
                for i, (eid, name) in enumerate(pending["candidates"].items())
            ]

            selected = await self.use("disambiguation_select", user_input, candidates=candidates)
            if not selected:
                _LOGGER.warning("[Stage1] Disambiguation selection empty")
                return {"status": "error", "result": await error_response(user_input)}

            intent_name = (pending.get("intent") or "").strip()
            # Pass through original params (e.g. brightness, position)
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

                conv_result = exec_data["result"]
                
                # Resume remaining commands if any
                remaining = pending.get("remaining")
                if remaining:
                    _LOGGER.debug("[Stage1] Resuming %d remaining commands.", len(remaining))
                    return await self._execute_sequence(user_input, remaining, previous_results=[conv_result])

                return {"status": "handled", "result": conv_result}

            except Exception as e:
                _LOGGER.exception("[Stage1] Direct action execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler.")}

        # --- Handle multiple entities from Stage0 --------------------------------
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            _LOGGER.debug("[Stage1] Multiple entities from Stage0 detected.")
            # Use unified logic
            return await self._process_candidates(
                user_input, 
                list(prev_result.resolved_ids), 
                (prev_result.intent or "").strip()
            )

        # --- Clarification: split into atomic commands -------------------------
        clar_data = await self.use("clarification", user_input)

        if isinstance(clar_data, list):
            original_norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clar_data if isinstance(c, str) and c.strip()]

            # Case 1: Clarification returned same text
            if len(atomic) == 1 and atomic[0].strip().lower() == original_norm:
                _LOGGER.debug("[Stage1] Clarification returned same text.")

                if isinstance(prev_result, Stage0Result) and prev_result.intent and prev_result.type == "intent":
                    return {"status": "escalate", "result": prev_result}

                ki_data = await self.use("keyword_intent", user_input) or {}
                intent_name = ki_data.get("intent")
                slots = ki_data.get("slots") or {}

                if not intent_name:
                    return {"status": "escalate", "result": prev_result}

                # Resolve
                er_data = await self.use("entity_resolver", user_input, entities=slots) or {}
                entity_ids = er_data.get("resolved_ids") or []

                # --- NEW: Area Alias Fallback ---
                if not entity_ids:
                    candidate_area = slots.get("area") or slots.get("name")
                    if candidate_area:
                        _LOGGER.debug("[Stage1] No entities found. Trying AreaAlias for '%s'", candidate_area)
                        alias_res = await self.use("area_alias", user_input, search_text=candidate_area)
                        mapped_area = alias_res.get("area")
                        
                        if mapped_area:
                            _LOGGER.debug("[Stage1] AreaAlias mapped '%s' → '%s'. Retrying resolution.", candidate_area, mapped_area)
                            new_slots = slots.copy()
                            new_slots["area"] = mapped_area
                            if new_slots.get("name") == candidate_area:
                                new_slots.pop("name") 
                            
                            er_data = await self.use("entity_resolver", user_input, entities=new_slots) or {}
                            entity_ids = er_data.get("resolved_ids") or []
                            slots = new_slots 
                # --------------------------------

                if not entity_ids:
                    return {"status": "escalate", "result": prev_result}

                # Execute (handling single OR multiple results via unified logic)
                params = {k: v for (k, v) in slots.items() if k not in ("name", "entity_id")}
                return await self._process_candidates(user_input, entity_ids, intent_name, params)

            # Case 2: Multi-command
            if len(atomic) > 1 or (len(atomic) == 1 and atomic[0].strip().lower() != original_norm):
                _LOGGER.debug("[Stage1] Multi-command detected.")
                return await self._execute_sequence(user_input, atomic)

        _LOGGER.debug("[Stage1] Escalate.")
        return {"status": "escalate", "result": prev_result}
