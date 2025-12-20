"""ExecutionPipeline: Orthogonal execution flow for all stages.

This module provides the unified execution pipeline that any stage can use
once intent + entities are resolved. The pipeline handles:

1. Knowledge Graph Device Coupling Filter
2. Smart State Filtering (turn off → only ON entities, etc.)
3. Plural Check ("alle" means no disambiguation)
4. Disambiguation (if needed)
5. Ensure KG Preconditions
6. Execute Intent
7. Verify State Change
8. Generate Confirmation
9. Handle Follow-up Questions
10. Store to Semantic User Cache (on success)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .capabilities.command_processor import CommandProcessorCapability
from .capabilities.semantic_cache import SemanticCacheCapability
from .stage_result import StageResult

_LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from execution pipeline."""
    success: bool
    response: Any  # ConversationResponse
    pending_data: Optional[Dict[str, Any]] = None  # For disambiguation follow-up


class ExecutionPipeline:
    """Unified execution pipeline for all stages.
    
    Once a stage resolves intent + entity_ids, it calls this pipeline
    to execute, verify, confirm, and cache the result.
    """

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self._processor = CommandProcessorCapability(hass, config)
        self._cache: Optional[SemanticCacheCapability] = None

    def set_cache(self, cache: SemanticCacheCapability):
        """Inject semantic cache for storing verified commands."""
        self._cache = cache
        self._processor.set_cache(cache)

    async def execute(
        self,
        user_input,
        stage_result: StageResult,
        from_cache: bool = False,
    ) -> ExecutionResult:
        """Execute resolved intent from any stage.
        
        Args:
            user_input: ConversationInput from Home Assistant
            stage_result: StageResult with status="success" containing intent + entities
            from_cache: If True, skip re-caching this command
            
        Returns:
            ExecutionResult with success status, response, and optional pending data
        """
        if stage_result.status != "success":
            raise ValueError(f"ExecutionPipeline requires status='success', got '{stage_result.status}'")

        if not stage_result.intent:
            raise ValueError("ExecutionPipeline requires intent to be set")

        if not stage_result.entity_ids:
            _LOGGER.warning("[ExecutionPipeline] No entity_ids provided, this may fail")

        # Delegate to existing CommandProcessor
        result = await self._processor.process(
            user_input=user_input,
            candidates=stage_result.entity_ids,
            intent_name=stage_result.intent,
            params=stage_result.params,
            learning_data=stage_result.context.get("learning_data"),
            from_cache=from_cache,
        )

        return ExecutionResult(
            success=result.get("status") == "handled",
            response=result.get("result"),
            pending_data=result.get("pending_data"),
        )

    async def continue_disambiguation(
        self,
        user_input,
        pending_data: Dict[str, Any],
    ) -> ExecutionResult:
        """Continue after user selects from disambiguation.
        
        Args:
            user_input: User's selection response
            pending_data: Stored disambiguation context
            
        Returns:
            ExecutionResult from executing the selected entity
        """
        result = await self._processor.continue_disambiguation(
            user_input=user_input,
            pending_data=pending_data,
        )

        return ExecutionResult(
            success=result.get("status") == "handled",
            response=result.get("result"),
            pending_data=result.get("pending_data"),
        )


# Singleton-ish factory for stages to share the pipeline
_pipeline_instance: Optional[ExecutionPipeline] = None


def get_execution_pipeline(hass, config) -> ExecutionPipeline:
    """Get or create the shared execution pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ExecutionPipeline(hass, config)
    return _pipeline_instance


def reset_execution_pipeline():
    """Reset the pipeline instance (for testing)."""
    global _pipeline_instance
    _pipeline_instance = None
