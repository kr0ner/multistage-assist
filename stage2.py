import logging
from typing import Any, Dict, List

from homeassistant.components import conversation
from homeassistant.helpers import intent as ha_intent

from .base_stage import BaseStage
from .capabilities.plural_detection import PluralDetectionCapability
from .conversation_utils import error_response
from .stage_result import Stage0Result

_LOGGER = logging.getLogger(__name__)


class Stage2Processor(BaseStage):
    name = "stage2"
    capabilities = [PluralDetectionCapability]

    def __init__(self, hass, config):
        super().__init__(hass, config)

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage2] Input='%s'", user_input.text)

        # Stage2 only acts when Stage0 already found multiple concrete targets.
        if not isinstance(prev_result, Stage0Result):
            _LOGGER.debug("[Stage2] No Stage0Result → escalate.")
            return {"status": "escalate", "result": prev_result}

        ids: List[str] = list(prev_result.resolved_ids or [])
        if len(ids) <= 1:
            _LOGGER.debug("[Stage2] <=1 resolved entity from Stage0 → nothing to do here, escalate.")
            return {"status": "escalate", "result": prev_result}

        # We have >1 entity candidates — now check if the user's phrasing really intended plural.
        pd = await self.use("plural_detection", user_input) or {}
        if pd.get("multiple_entities") is not True:
            _LOGGER.debug("[Stage2] Plural not confirmed → escalate for disambiguation or later handling.")
            return {"status": "escalate", "result": prev_result}

        # Plural confirmed: execute the same Hass intent for all resolved entities directly,
        # without re-running the NLU or rebuilding the intent.
        intent_name = (prev_result.intent or "").strip()
        original_text = prev_result.raw or user_input.text

        try:
            _LOGGER.debug("[Stage2] Plural confirmed → executing '%s' for %d entities.", intent_name, len(ids))
            responses = []
            for eid in ids:
                slots = {"name": {"value": eid}}
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

            # Prefer a response that already contains plain speech; else fall back to last response.
            def _has_plain_speech(r) -> bool:
                s = getattr(r, "speech", None)
                return bool(isinstance(s, dict) and (s.get("plain") or {}).get("speech"))

            final_resp = next((r for r in reversed(responses) if _has_plain_speech(r)), responses[-1])

            # Wrap in ConversationResult and finish.
            conv_result = conversation.ConversationResult(
                response=final_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
            _LOGGER.debug("[Stage2] Collective execution done.")
            return {"status": "handled", "result": conv_result}

        except Exception as e:
            _LOGGER.exception("[Stage2] Plural execution failed: %s", e)
            return {"status": "error", "result": await error_response(user_input, "Fehler beim Ausführen des Befehls.")}
