import logging
from typing import Any, Dict
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
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

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
            original_text = pending.get("raw") or user_input.text

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
                return {"status": "handled", "result": conv_result}

            except Exception as e:
                _LOGGER.exception("[Stage1] Direct action execution failed: %s", e)
                return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

        # --- Handle multiple entities from Stage0 --------------------------------
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            _LOGGER.debug("[Stage1] Multiple entities from Stage0 detected → checking plurality first.")

            # 1) Plural detection FIRST: if user clearly meant multiple, execute directly for all
            pd = await self.use("plural_detection", user_input) or {}
            if pd.get("multiple_entities") is True:
                _LOGGER.debug("[Stage1] Plural confirmed → executing action for all resolved entities (no disambiguation).")

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
                        _LOGGER.warning("[Stage1] IntentExecutorCapability returned no result for plural execution.")
                        return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

                    conv_result = exec_data["result"]
                    final_resp = conv_result.response

                    # Speech fallback via confirmation if needed
                    def _has_plain_speech(r) -> bool:
                        s = getattr(r, "speech", None)
                        if not s or not isinstance(s, dict):
                            return False
                        plain = s.get("plain") or {}
                        return bool(plain.get("speech"))

                    if not _has_plain_speech(final_resp):
                        friendly = []
                        for eid in entities:
                            st = self.hass.states.get(eid)
                            name = (st and st.attributes.get("friendly_name")) or eid
                            friendly.append({"entity_id": eid, "name": name})

                        conf = await self.use(
                            "intent_confirmation",
                            user_input,
                            intent=intent_name,
                            entities=friendly,
                            params={},
                            language=user_input.language or "de",
                            style="concise",
                        )
                        msg = (conf or {}).get("message")
                        if msg:
                            final_resp.async_set_speech(msg)

                    _LOGGER.debug("[Stage1] Multi-target execution completed without disambiguation.")
                    return {"status": "handled", "result": conv_result}

                except Exception as e:
                    _LOGGER.exception("[Stage1] Direct multi-target execution failed: %s", e)
                    return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

            # 2) Otherwise: NOT clearly plural → ask disambiguation like before
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

                # That should never happen
                if isinstance(prev_result, Stage0Result) and prev_result.intent and prev_result.type == "intent":
                    _LOGGER.debug("[Stage1] Stage0 already has a known intent → escalate with prev_result.")
                    return {"status": "escalate", "result": prev_result}

                # Sonst: KeywordIntentCapability fragen
                ki_data = await self.use("keyword_intent", user_input) or {}
                intent_name = ki_data.get("intent")
                slots = ki_data.get("slots") or {}

                if not intent_name:
                    _LOGGER.debug("[Stage1] KeywordIntentCapability could not derive an intent → escalate to next stage.")
                    return {"status": "escalate", "result": prev_result}

                # Derive entities from slots
                er_data = await self.use("entity_resolver", user_input, entities=slots) or {}
                entity_ids = er_data.get("resolved_ids") or []

                if not entity_ids:
                    _LOGGER.debug(
                        "[Stage1] EntityResolver could not resolve any entities for derived intent '%s' (slots=%s)",
                        intent_name,
                        slots,
                    )
                    # Escalate instead of guessing
                    return {"status": "escalate", "result": prev_result}

                # Params für IntentExecutor: alle Slots außer 'name' (wird vom Executor gesetzt) und 'entity_id'
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
                except Exception as e:
                    _LOGGER.exception("[Stage1] Direct execution of derived intent failed: %s", e)
                    return {
                        "status": "error",
                        "result": await error_response(user_input, "Fehler beim Ausführen des Befehls."),
                    }

                if not exec_data or "result" not in exec_data:
                    _LOGGER.warning("[Stage1] IntentExecutorCapability returned no result for derived intent.")
                    return {
                        "status": "error",
                        "result": await error_response(user_input, "Fehler beim Ausführen des Befehls."),
                    }

                _LOGGER.debug(
                    "[Stage1] Successfully executed derived intent '%s' for %d entities",
                    intent_name,
                    len(entity_ids),
                )
                return {"status": "handled", "result": exec_data["result"]}

            # Case 2: LLM produced list > 1 of atomic results
            if len(atomic) > 1 or (len(atomic) == 1 and atomic[0].strip().lower() != original_norm):
                _LOGGER.debug("[Stage1] Clarification detected multiple/changed atomic commands → executing each via pipeline.")

                agent = self.agent  # MultiStageAssistAgent
                results = []
                for i, cmd in enumerate(atomic, start=1):
                    _LOGGER.debug("[Stage1] Executing atomic command %d/%d: %s", i, len(atomic), cmd)
                    sub_input = with_new_text(user_input, cmd)
                    result = await agent._run_pipeline(sub_input)
                    results.append(result)

                if results:
                    _LOGGER.debug("[Stage1] Multi-command execution finished (%d commands)", len(results))
                    return {"status": "handled", "result": results[-1]}

                _LOGGER.warning("[Stage1] No valid atomic commands executed.")
                return {
                    "status": "error",
                    "result": await error_response(user_input, "Keine gültigen Befehle erkannt."),
                }

        # --- Default: no clarification, no disambiguation ----------------------
        _LOGGER.debug("[Stage1] No clarification or disambiguation triggered → escalate to next stage.")
        return {"status": "escalate", "result": prev_result}
