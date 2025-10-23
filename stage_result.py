import logging
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class Stage0Result:
    """Container for Stage0 parsed intent and resolved entities."""

    def __init__(
        self,
        type: str,
        intent: Optional[str] = None,
        raw: Optional[str] = None,
        resolved_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.type = type
        self.intent = intent
        self.raw = raw
        self.resolved_ids = resolved_ids or []
        self.extra = kwargs or {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "intent": self.intent,
            "raw": self.raw,
            "resolved_ids": self.resolved_ids,
            **self.extra,
        }

    def __repr__(self):
        return (
            f"<Stage0Result type={self.type!r} intent={self.intent!r} "
            f"resolved={len(self.resolved_ids)}>"
        )
