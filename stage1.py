import logging
from typing import Any, Dict
from .base_stage import BaseStage
from .capabilities.clarification import ClarificationCapability
from .capabilities.disambiguation import DisambiguationCapability
from .capabilities.disambiguation_select import DisambiguationSelectCapability
from .capabilities.plural_detection import PluralDetectionCapability
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
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s', prev_result=%s", user_input.text, type(prev_result).__name__)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # --- Handle disambiguation follow-up (now using disambiguation_select) ---
        if key in self._pending:
            _LOGGER.debug("[Stage1] Resuming pending disambiguation for key=%s", key)
            pending = self._pending.pop(key, None)
            if not pending:
                _LOGGER.warning("[Stage1] Pending state lost for key=%s", key)
                return {"status": "error", "result": await error_response(user_input)}

            # --- Build candidates with ordinals (1-based)
            candidates = [
                {"entity_id": eid, "name": name, "ordinal": i + 1}
                for i, (eid, name) in enumerate(pending["candidates"].items())
            ]

            selected = await self.use("disambiguation_select", user_input, candidates=candidates)
            if not selected:
                _LOGGER.warning("[Stage1] Disambiguation selection empty for input='%s'", user_input.text)
                return {"status": "error", "result": await error_response(user_input)}

            _LOGGER.debug("[Stage1] Disambiguation selected entities=%s", selected)

            # --- Patch the original intent from pending context
            patched_result = Stage0Result(
                type="intent",
                intent=pending.get("intent"),
                raw=pending.get("raw") or user_input.text,
                resolved_ids=selected,
            )

            # --- Re-run the pipeline as if the user just said the original command
            _LOGGER.debug("[Stage1] Reinvoking full pipeline with patched Stage0Result: %s", patched_result.intent)
            agent = self.hass.data.get("custom_components.multistage_assist_agent")
            if not agent:
                _LOGGER.error("[Stage1] MultiStageAssistAgent not registered in hass.data")
                return {"status": "error", "result": await error_response(user_input)}

            # Reuse the same pipeline logic
            result = await agent._run_pipeline(with_new_text(user_input, pending.get("raw", user_input.text)), prev_result=patched_result)
            return {"status": "handled", "result": result.get("result") if isinstance(result, dict) else result}

        # --- Handle multiple entities from Stage0 (ask a question) -------------
        if isinstance(prev_result, Stage0Result) and len(prev_result.resolved_ids or []) > 1:
            _LOGGER.debug("[Stage1] Multiple entities detected → initiating disambiguation.")
            entities = {
                eid: (self.hass.states.get(eid).attributes.get("friendly_name", eid) if self.hass.states.get(eid) else eid)
                for eid in prev_result.resolved_ids
            }
            data = await self.use("disambiguation", user_input, entities=entities)
            msg = (data or {}).get("message") or "Welches Gerät meinst du?"

            # store original 'raw' so we can preserve the user's original text later
            self._pending[key] = {"candidates": entities, "intent": prev_result.intent, "raw": prev_result.raw}
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
