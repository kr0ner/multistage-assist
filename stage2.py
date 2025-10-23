import logging
from typing import Any, Dict
from .base_stage import BaseStage
from .capabilities.disambiguation_resolution import DisambiguationResolutionCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .conversation_utils import make_response, error_response

_LOGGER = logging.getLogger(__name__)


class Stage2Processor(BaseStage):
    name = "stage2"
    capabilities = [
        DisambiguationResolutionCapability,
        PluralDetectionCapability,
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage2] Running advanced reasoning for input='%s'", user_input.text)

        key = getattr(user_input, "session_id", None) or user_input.conversation_id
        if key in self._pending:
            _LOGGER.debug("[Stage2] Handling pending disambiguation follow-up.")
            pending = self._pending.pop(key)
            candidates = [{"entity_id": k, "name": v} for k, v in pending.get("candidates", {}).items()]
            result = await self.use("disambiguation_resolution", user_input, candidates=candidates)
            if not result or not result.get("entities"):
                return {"status": "error", "result": await error_response(user_input, "Ich habe das nicht verstanden.")}
            return {"status": "handled", "result": await make_response(result.get("message"), user_input)}

        # Example: Detect plural and act accordingly
        plural_info = await self.use("plural_detection", user_input)
        if plural_info and plural_info.get("multiple_entities"):
            _LOGGER.debug("[Stage2] Detected plural entities → executing collective intent.")
            # You can execute all lights or sensors here
            return {"status": "handled", "result": await make_response("Alles klar, ich kümmere mich darum.", user_input)}

        _LOGGER.debug("[Stage2] No special handling → passing upward or finalizing.")
        return {"status": "handled", "result": await make_response("Ich habe das ausgeführt.", user_input)}
