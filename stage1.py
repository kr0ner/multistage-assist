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
                responses = []
                for eid in selected:
                    slots = {"name": {"value": eid}}
                    _LOGGER.debug("[Stage1] Executing action intent '%s' with slots=%s", intent_name, slots)
                    resp = await ha_intent.async_handle(
                        self.hass,
                        platform="conversation",
                        intent_type=intent_name,
                        slots=slots,
                        text_input=original_text,
                        context=user_input.context,
                        language=user_input.language or "de",
                    )
                    responses.append(resp)

                # Helper to check if a response already contains plain speech
                def _has_plain_speech(r) -> bool:
                    s = getattr(r, "speech", None)
                    if not s or not isinstance(s, dict):
                        return False
                    plain = s.get("plain") or {}
                    return bool(plain.get("speech"))

                # Prefer the last response that already has speech; else fall back to the last response
                final_resp = next((r for r in reversed(responses) if _has_plain_speech(r)), responses[-1])

                # If no handler produced speech, synthesize a concise confirmation via capability
                if not _has_plain_speech(final_resp):
                    # Build friendly list for confirmation (entity_id + display name)
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

                conv_result = conversation.ConversationResult(
                    response=final_resp,
                    conversation_id=user_input.conversation_id,
                    continue_conversation=False,
                )
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
                original_text = prev_result.raw or user_input.text
                entities = list(prev_result.resolved_ids)

                try:
                    responses = []
                    for eid in entities:
                        slots = {"name": {"value": eid}}
                        _LOGGER.debug("[Stage1] Executing action intent '%s' with slots=%s", intent_name, slots)
                        resp = await ha_intent.async_handle(
                            self.hass,
                            platform="conversation",
                            intent_type=intent_name,
                            slots=slots,
                            text_input=original_text,
                            context=user_input.context,
                            language=user_input.language or "de",
                        )
                        responses.append(resp)

                    def _has_plain_speech(r) -> bool:
                        s = getattr(r, "speech", None)
                        if not s or not isinstance(s, dict):
                            return False
                        plain = s.get("plain") or {}
                        return bool(plain.get("speech"))

                    # Prefer a response that already has speech; else last response
                    final_resp = next((r for r in reversed(responses) if _has_plain_speech(r)), responses[-1])

                    # If no handler produced speech, synthesize a concise confirmation
                    if not _has_plain_speech(final_resp):
                        # Build friendly list for confirmation (entity_id + display name)
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
                            params={},  # keep generic; Stage1 doesn't invent params
                            language=user_input.language or "de",
                            style="concise",
                        )
                        msg = (conf or {}).get("message")
                        if msg:
                            final_resp.async_set_speech(msg)

                    conv_result = conversation.ConversationResult(
                        response=final_resp,
                        conversation_id=user_input.conversation_id,
                        continue_conversation=False,
                    )
                    _LOGGER.debug("[Stage1] Multi-target execution completed without disambiguation.")
                    return {"status": "handled", "result": conv_result}

                except Exception as e:
                    _LOGGER.exception("[Stage1] Direct multi-target execution failed: %s", e)
                    return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}

            # 2) Otherwise: NOT clearly plural → ask disambiguation like before
            _LOGGER.debug("[Stage1] Plural not confirmed → initiating disambiguation.")
            entities_map = {}
            for eid in prev_result.resolved_ids:
                st = self.hass.states.get(eid)
                name = (st and st.attributes.get("friendly_name")) or eid
                entities_map[eid] = name

            data = await self.use("disambiguation", user_input, entities=entities_map)
            msg = (data or {}).get("message") or "Welches Gerät meinst du?"

            # store original 'raw' to preserve the user's original text later
            self._pending[key] = {"candidates": entities_map, "intent": prev_result.intent, "raw": prev_result.raw}
            _LOGGER.debug("[Stage1] Stored pending disambiguation context for %s", key)
            return {"status": "handled", "result": await make_response(msg, user_input)}

        # --- Clarification & multi-command parsing -----------------------------
        clar_data = await self.use("clarification", user_input)

        if isinstance(clar_data, list):
            _LOGGER.debug("[Stage1] Clarification produced %d atomic commands", len(clar_data))

            if len(clar_data) == 1 and clar_data[0].strip().lower() == (user_input.text or "").strip().lower():
                _LOGGER.debug("[Stage1] Clarification returned the same text → escalate forward.")
                return {"status": "escalate", "result": prev_result}

            agent = self.hass.data.get("custom_components.multistage_assist_agent")
            if not agent:
                _LOGGER.error("[Stage1] MultiStageAssistAgent not registered in hass.data")
                return {"status": "error", "result": await error_response(user_input)}

            results = []
            for i, cmd in enumerate(clar_data, start=1):
                if cmd.strip().lower() == (user_input.text or "").strip().lower():
                    _LOGGER.debug("[Stage1] Command #%d identical to input → skipping to avoid loop.", i)
                    continue
                _LOGGER.debug("[Stage1] Executing atomic command %d/%d: %s", i, len(clar_data), cmd)
                sub_input = with_new_text(user_input, cmd)
                result = await agent._run_pipeline(sub_input)
                results.append(result)

            if results:
                _LOGGER.debug("[Stage1] Multi-command execution finished (%d commands)", len(results))
                return {"status": "handled", "result": results[-1]}

            _LOGGER.warning("[Stage1] No valid atomic commands executed.")
            return {"status": "error", "result": await error_response(user_input, "Keine gültigen Befehle erkannt.")}

        # --- Default: no clarification, no disambiguation ----------------------
        _LOGGER.debug("[Stage1] No clarification or disambiguation triggered → escalate to next stage.")
        return {"status": "escalate", "result": prev_result}
