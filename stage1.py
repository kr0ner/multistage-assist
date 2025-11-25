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
            
            # recursive call to pipeline
            result = await agent._run_pipeline(sub_input)
            
            # Check if this command triggered a pending state (disambiguation needed)
            if key in self._pending:
                _LOGGER.debug("[Stage1] Command '%s' triggered pending state. Halting sequence.", cmd)
                
                # Store remaining commands for later resumption
                remaining = commands[i+1:]
                if remaining:
                    self._pending[key]["remaining"] = remaining
                    _LOGGER.debug("[Stage1] Stored %d remaining commands for later.", len(remaining))
                
                # Merge previous success messages into the question/result so user knows what happened so far
                if results:
                    self._merge_speech(result, results)
                
                return {"status": "handled", "result": result}
            
            results.append(result)

        if not results:
             return {"status": "error", "result": await error_response(user_input, "Keine Befehle ausgeführt.")}
        
        final = results[-1]
        # Merge all previous results into the final one
        self._merge_speech(final, results[:-1])
        _LOGGER.debug("[Stage1] Sequence execution finished. Merged %d results.", len(results))
        return {"status": "handled", "result": final}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s', prev_result=%s", user_input.text, type(prev_result).__name__)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # --- Handle disambiguation follow-up: execute the same intent with patched targets ---
        if key in self._pending:
            _LOGGER.debug("[Stage1] Resuming pending disambiguation for key=%s", key)
            pending = self._pending.pop(key, None)
            if not pending:
                _LOGGER.warning("[Stage1] Pending state lost for key=%s", key)
                return {"status": "error", "result": await error_response(user_input)}

            # 1-based ordinals for "das erste/zweite/letzte"
            candidates = [
                {"entity_id": eid, "name": name, "ordinal": i + 1}
                for i, (eid, name) in enumerate(pending["candidates"].items())
            ]

            selected = await self.use("disambiguation_select", user_input, candidates=candidates)
            if not selected:
                _LOGGER.warning("[Stage1] Disambiguation selection empty for input='%s'", user_input.text)
                return {"status": "error", "result": await error_response(user_input)}

            _LOGGER.debug("[Stage1] Disambiguation selected entities=%s", selected)

            intent_name = (pending.get("intent") or "").strip()

            try:
                # Execute patched intent (only for selected entities) via capability
                exec_data = await self.use(
                    "intent_executor",
                    user_input,
                    intent_name=intent_name,
                    entity_ids=selected,
                    language=user_input.language or "de",
                )
                if not exec_data or "result" not in exec_data:
                    _LOGGER.warning("[Stage1] IntentExecutorCapability returned no result.")
                    return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

                conv_result = exec_data["result"]
                final_resp = conv_result.response

                # If no handler produced speech, synthesize concise confirmation via capability
                def _has_plain_speech(r) -> bool:
                    s = getattr(r, "speech", None)
                    if not s or not isinstance(s, dict):
                        return False
                    plain = s.get("plain") or {}
                    return bool(plain.get("speech"))

                if not _has_plain_speech(final_resp):
                    friendly = []
                    for eid in selected:
                        name = pending["candidates"].get(eid)
                        if not name:
                            st = self.hass.states.get(eid)
                            name = (st and st.attributes.get("friendly_name")) or eid
                        friendly.append({"entity_id": eid, "name": name})

                    conf = await self.use(
                        "intent_confirmation",
                        user_input,
                        intent=intent_name,
                        entities=friendly,
                        params=pending.get("params", {}) or {},
                        language=user_input.language or "de",
                        style="concise",
                    )
                    msg = (conf or {}).get("message")
                    if msg:
                        final_resp.async_set_speech(msg)

                _LOGGER.debug("[Stage1] Action disambiguation resolved and executed successfully")

                # Check for remaining commands to resume
                remaining = pending.get("remaining")
                if remaining:
                    _LOGGER.debug("[Stage1] Resuming %d remaining commands after disambiguation.", len(remaining))
                    return await self._execute_sequence(user_input, remaining, previous_results=[conv_result])

                return {"status": "handled", "result": conv_result}

            except Exception as e:
                _LOGGER.exception("[Stage1] Direct action execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

        # --- Handle multiple entities from Stage0 --------------------------------
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            _LOGGER.debug("[Stage1] Multiple entities from Stage0 detected → checking plurality first.")

            # 1) Plural detection FIRST
            pd = await self.use("plural_detection", user_input) or {}
            if pd.get("multiple_entities") is True:
                _LOGGER.debug("[Stage1] Plural confirmed → executing action for all resolved entities.")
                
                intent_name = (prev_result.intent or "").strip()
                entities = list(prev_result.resolved_ids)

                try:
                    exec_data = await self.use(
                        "intent_executor",
                        user_input,
                        intent_name=intent_name,
                        entity_ids=entities,
                        language=user_input.language or "de",
                    )
                    if not exec_data or "result" not in exec_data:
                        return {"status": "error", "result": await error_response(user_input, "Fehler.")}

                    conv_result = exec_data["result"]
                    # (Speech logic omitted for brevity, handled by executor generally)
                    return {"status": "handled", "result": conv_result}

                except Exception as e:
                    _LOGGER.exception("[Stage1] Direct multi-target execution failed: %s", e)
                    return {"status": "error", "result": await error_response(user_input, "Fehler.")}

            # 2) Otherwise: Disambiguation
            _LOGGER.debug("[Stage1] Plural not confirmed → initiating disambiguation.")
            entities_map = {
                eid: self.hass.states.get(eid).attributes.get("friendly_name", eid)
                for eid in prev_result.resolved_ids
            }
            data = await self.use("disambiguation", user_input, entities=entities_map)
            msg = (data or {}).get("message") or "Welches Gerät meinst du?"

            # store original 'raw' to preserve the user's original text later
            self._pending[key] = {"candidates": entities_map, "intent": prev_result.intent, "raw": prev_result.raw}
            _LOGGER.debug("[Stage1] Stored pending disambiguation context for %s", key)
            return {"status": "handled", "result": await make_response(msg, user_input)}

        # --- Clarification: split into atomic commands -------------------------
        clar_data = await self.use("clarification", user_input)

        if isinstance(clar_data, list):
            _LOGGER.debug("[Stage1] Clarification produced %d atomic commands", len(clar_data))

            # Normalize
            original_norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clar_data if isinstance(c, str) and c.strip()]

            # Case 1: Clarification returned same text
            if len(atomic) == 1 and atomic[0].strip().lower() == original_norm:
                _LOGGER.debug("[Stage1] Clarification returned the same text → try keyword-based intent derivation.")

                if isinstance(prev_result, Stage0Result) and prev_result.intent and prev_result.type == "intent":
                    _LOGGER.debug("[Stage1] Stage0 already has a known intent → escalate with prev_result.")
                    return {"status": "escalate", "result": prev_result}

                ki_data = await self.use("keyword_intent", user_input) or {}
                intent_name = ki_data.get("intent")
                slots = ki_data.get("slots") or {}

                if not intent_name:
                    _LOGGER.debug("[Stage1] KeywordIntentCapability could not derive an intent → escalate.")
                    return {"status": "escalate", "result": prev_result}

                er_data = await self.use("entity_resolver", user_input, entities=slots) or {}
                entity_ids = er_data.get("resolved_ids") or []

                if not entity_ids:
                    _LOGGER.debug("[Stage1] EntityResolver could not resolve any entities for derived intent.")
                    return {"status": "escalate", "result": prev_result}

                params = {k: v for (k, v) in slots.items() if k not in ("name", "entity_id")}

                try:
                    exec_data = await self.use(
                        "intent_executor",
                        user_input,
                        intent_name=intent_name,
                        entity_ids=entity_ids,
                        params=params,
                        language=user_input.language or "de",
                    )
                    if not exec_data or "result" not in exec_data:
                         return {"status": "error", "result": await error_response(user_input, "Fehler.")}
                    
                    return {"status": "handled", "result": exec_data["result"]}
                except Exception as e:
                    _LOGGER.exception("[Stage1] execution failed: %s", e)
                    return {"status": "error", "result": await error_response(user_input, "Fehler.")}

            # Case 2: LLM produced list > 1 of atomic results
            if len(atomic) > 1 or (len(atomic) == 1 and atomic[0].strip().lower() != original_norm):
                _LOGGER.debug("[Stage1] Clarification detected multiple/changed atomic commands → executing each via pipeline.")
                return await self._execute_sequence(user_input, atomic)

        # --- Default: no clarification, no disambiguation ----------------------
        _LOGGER.debug("[Stage1] No clarification or disambiguation triggered → escalate to next stage.")
        return {"status": "escalate", "result": prev_result}
