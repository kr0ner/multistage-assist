"""Unified stage result types for the multi-stage pipeline.

All stages return StageResult with a common interface, enabling
consistent flow control and context passing between stages.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Unified result container for all processing stages.
    
    Attributes:
        status: Processing outcome
            - "success": Intent + entities resolved → execute via ExecutionPipeline
            - "escalate": Need more sophisticated processing → next stage
            - "escalate_chat": User wants to chat → Stage3 in chat-only mode
            - "pending": Asking user for clarification (area, entity, etc.) → wait for response
            - "error": Unrecoverable error
        intent: Derived intent name (e.g., "HassTurnOn", "HassLightSet")
        entity_ids: Resolved entity IDs to act upon
        params: Intent parameters (brightness, temperature, duration, etc.)
        context: Enriched context for next stage (NLU hints, slot values)
        response: Pre-built response for error/pending cases
        pending_data: Data for pending multi-turn (type, candidates, etc.)
        raw_text: Original user utterance (for logging/learning)
    """
    
    status: Literal["success", "escalate", "escalate_chat", "multi_command", "pending", "error"]
    intent: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    response: Optional[Any] = None  # ConversationResponse or similar
    pending_data: Dict[str, Any] = field(default_factory=dict)  # For pending status
    raw_text: Optional[str] = None
    commands: List[str] = field(default_factory=list)  # For multi_command status

    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging."""
        return {
            "status": self.status,
            "intent": self.intent,
            "entity_ids": self.entity_ids,
            "params": self.params,
            "context": self.context,
            "raw_text": self.raw_text,
        }
    
    @classmethod
    def success(
        cls,
        intent: str,
        entity_ids: List[str],
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Create a success result ready for execution."""
        return cls(
            status="success",
            intent=intent,
            entity_ids=entity_ids,
            params=params or {},
            context=context or {},
            raw_text=raw_text,
        )
    
    @classmethod
    def escalate(
        cls,
        context: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Create an escalation result to pass to the next stage."""
        return cls(
            status="escalate",
            context=context or {},
            raw_text=raw_text,
        )
    
    @classmethod
    def escalate_chat(
        cls,
        context: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Escalate to chat mode (Stage3 cloud with minimal context)."""
        ctx = context or {}
        ctx["chat_mode"] = True
        return cls(
            status="escalate_chat",
            context=ctx,
            raw_text=raw_text,
        )
    
    @classmethod
    def error(
        cls,
        response: Any,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Create an error result with pre-built response."""
        return cls(
            status="error",
            response=response,
            raw_text=raw_text,
        )
    
    @classmethod
    def multi_command(
        cls,
        commands: List[str],
        context: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Create a multi-command result for processing multiple atomic commands."""
        return cls(
            status="multi_command",
            commands=commands,
            context=context or {},
            raw_text=raw_text,
        )

    @classmethod
    def pending(
        cls,
        pending_type: str,
        message: str,
        pending_data: Optional[Dict[str, Any]] = None,
        raw_text: Optional[str] = None,
    ) -> "StageResult":
        """Create a pending result for multi-turn user interaction.
        
        Args:
            pending_type: Type of pending state (area_learning, disambiguation, etc.)
            message: Question to ask the user
            pending_data: Context for continuing when user responds
            raw_text: Original user utterance
        """
        data = pending_data or {}
        data["type"] = pending_type
        data["original_prompt"] = message
        return cls(
            status="pending",
            pending_data=data,
            raw_text=raw_text,
        )


