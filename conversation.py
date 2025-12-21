"""Multi-Stage Assist conversation agent.

This orchestrator runs through stages sequentially:
- Stage0: NLU (built-in Home Assistant)
- Stage1: Semantic Cache (fast path from cached commands)
- Stage2: Local LLM (Ollama keyword_intent)
- Stage3: Gemini Cloud (fallback + chat)

Each stage returns a StageResult. On "success", we execute via ExecutionPipeline.
On "escalate", we pass context to the next stage.
"""

import logging
from typing import Any, List, Optional

from homeassistant.components import conversation

from .stage0 import Stage0Processor
from .stage1_cache import Stage1CacheProcessor
from .stage2_llm import Stage2LLMProcessor
from .stage3_gemini import Stage3GeminiProcessor
from .stage_result import StageResult
from .execution_pipeline import ExecutionPipeline

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Dynamic 4-stage orchestrator for Home Assistant Assist."""

    def __init__(self, hass, config):
        self.hass = hass
        self.hass.data["custom_components.multistage_assist_agent"] = self
        self.config = config
        
        # Initialize 4-stage pipeline
        _LOGGER.info("[MultiStageAssist] Initializing 4-stage pipeline")
        self.stages: List[Any] = [
            Stage0Processor(hass, config),
            Stage1CacheProcessor(hass, config),
            Stage2LLMProcessor(hass, config),
            Stage3GeminiProcessor(hass, config),
        ]
        
        # Give every stage a back-reference to the orchestrator
        for stage in self.stages:
            stage.agent = self
        
        # Create execution pipeline
        self._execution_pipeline = ExecutionPipeline(hass, config)
        
        # Inject semantic cache into execution pipeline if available
        stage1 = self.stages[1]
        if hasattr(stage1, 'has') and stage1.has("semantic_cache"):
            cache = stage1.get("semantic_cache")
            self._execution_pipeline.set_cache(cache)

    @property
    def supported_languages(self) -> set[str]:
        return {"de"}

    async def _fallback(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        """Single place to hit the default HA agent."""
        return await conversation.async_converse(
            self.hass,
            text=user_input.text,
            context=user_input.context,
            conversation_id=user_input.conversation_id,
            language=user_input.language or "de",
            agent_id=conversation.HOME_ASSISTANT_AGENT,
        )

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        _LOGGER.info("Received utterance: %s", user_input.text)

        # If any stage owns a pending turn, let it resolve first.
        for stage in self.stages:
            if hasattr(stage, "has_pending") and stage.has_pending(user_input):
                _LOGGER.debug("Resuming pending interaction in %s", stage.__class__.__name__)
                pending = await stage.resolve_pending(user_input)
                if not pending:
                    _LOGGER.warning("%s returned None on pending resolution", stage.__class__.__name__)
                    break

                status, value = pending.get("status"), pending.get("result")
                if status == "handled":
                    return value or await self._fallback(user_input)
                if status == "error":
                    return value or await self._fallback(user_input)
                if status == "escalate":
                    return await self._run_pipeline(user_input, value)

                _LOGGER.warning("Unexpected pending format from %s: %s", stage.__class__.__name__, pending)

        # Run pipeline
        result = await self._run_pipeline(user_input)
        return result or await self._fallback(user_input)

    async def _run_pipeline(
        self, 
        user_input: conversation.ConversationInput, 
        context: Optional[dict] = None
    ) -> Optional[conversation.ConversationResult]:
        """Run unified pipeline with StageResult interface."""
        current_context = context or {}
        
        for stage in self.stages:
            try:
                result: StageResult = await stage.process(user_input, current_context)
            except Exception:
                _LOGGER.exception("%s.process() failed", stage.__class__.__name__)
                continue

            _LOGGER.debug(
                "[Pipeline] %s returned status=%s", 
                stage.__class__.__name__, result.status
            )

            if result.status == "success":
                # Execute via unified pipeline
                if result.intent:
                    # Only update cache for Stage2/Stage3 resolutions (new learnings)
                    # Skip cache update for:
                    # - from_cache: already in cache
                    # - from_nlu: Stage0 NLU handled it (builtin HA patterns)
                    # - Early stages (Stage0, Stage1) don't produce new learnings
                    skip_cache = (
                        result.context.get("from_cache", False) or
                        result.context.get("from_nlu", False) or
                        stage.__class__.__name__ in ("Stage0Processor", "Stage1CacheProcessor")
                    )
                    exec_result = await self._execution_pipeline.execute(
                        user_input,
                        result,
                        from_cache=skip_cache,
                    )
                    if exec_result.success:
                        return exec_result.response
                    else:
                        _LOGGER.warning("[Pipeline] Execution failed")
                        return exec_result.response
                elif result.response:
                    # Chat response (no intent)
                    return result.response
                    
            elif result.status == "escalate":
                # Pass enriched context to next stage
                current_context = {**current_context, **result.context}
                continue
                
            elif result.status == "escalate_chat":
                # Fast-track to chat mode
                current_context = {**current_context, **result.context}
                continue
                
            elif result.status == "error":
                if result.response:
                    return result.response
                continue

        _LOGGER.warning("All stages exhausted without result")
        return None
