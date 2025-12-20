"""Multi-Stage Assist conversation agent.

This orchestrator runs through stages sequentially:
- Stage0: NLU (built-in Home Assistant)
- Stage1: Semantic Cache (optional, fast path)
- Stage2: Local LLM (keyword_intent)
- Stage3: Gemini Cloud (fallback + chat)

Each stage returns a StageResult. On "success", we execute via ExecutionPipeline.
On "escalate", we pass context to the next stage.
"""

import logging
from typing import Any, List, Optional

from homeassistant.components import conversation

from .stage0 import Stage0Processor
from .stage1 import Stage1Processor  # Legacy - still used for full functionality
from .stage2 import Stage2Processor  # Legacy chat
from .stage1_cache import Stage1CacheProcessor  # New: cache-only
from .stage2_llm import Stage2LLMProcessor      # New: LLM-only
from .stage3_gemini import Stage3GeminiProcessor  # New: Gemini cloud
from .stage_result import StageResult
from .execution_pipeline import ExecutionPipeline

_LOGGER = logging.getLogger(__name__)


class MultiStageAssistAgent(conversation.AbstractConversationAgent):
    """Dynamic N-stage orchestrator for Home Assistant Assist."""

    def __init__(self, hass, config):
        self.hass = hass
        self.hass.data["custom_components.multistage_assist_agent"] = self
        self.config = config
        
        # Determine which pipeline to use
        use_new_pipeline = config.get("use_new_pipeline", False)
        
        if use_new_pipeline:
            # NEW: Unified pipeline with process() interface
            _LOGGER.info("[MultiStageAssist] Using NEW unified pipeline")
            self.stages: List[Any] = [
                Stage0Processor(hass, config),
                Stage1CacheProcessor(hass, config),
                Stage2LLMProcessor(hass, config),
                Stage3GeminiProcessor(hass, config),
            ]
            self._use_new_pipeline = True
        else:
            # LEGACY: Old pipeline with run() interface
            _LOGGER.info("[MultiStageAssist] Using LEGACY pipeline")
            self.stages: List[Any] = [
                Stage0Processor(hass, config),
                Stage1Processor(hass, config),
                Stage2Processor(hass, config),
            ]
            self._use_new_pipeline = False
        
        # Give every stage a back-reference to the orchestrator
        for stage in self.stages:
            stage.agent = self
        
        # Create execution pipeline for new stages
        self._execution_pipeline = ExecutionPipeline(hass, config)
        
        # Inject semantic cache into execution pipeline if available
        if len(self.stages) > 1:
            legacy_stage = self.stages[1]
            if hasattr(legacy_stage, 'has') and legacy_stage.has("semantic_cache"):
                cache = legacy_stage.get("semantic_cache")
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

        # Fresh pipeline
        if self._use_new_pipeline:
            result = await self._run_unified_pipeline(user_input)
        else:
            result = await self._run_pipeline(user_input)
        return result or await self._fallback(user_input)

    async def _run_unified_pipeline(
        self, 
        user_input: conversation.ConversationInput, 
        context: Optional[dict] = None
    ) -> Optional[conversation.ConversationResult]:
        """NEW: Run unified pipeline with StageResult interface."""
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
                    exec_result = await self._execution_pipeline.execute(
                        user_input,
                        result,
                        from_cache=result.context.get("from_cache", False),
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

    async def _run_pipeline(self, user_input: conversation.ConversationInput, prev_result: Any = None):
        """LEGACY: Run pipeline with dict-based interface."""
        current = prev_result
        for stage in self.stages:
            try:
                out = await stage.run(user_input, current)
            except Exception:
                _LOGGER.exception("%s failed", stage.__class__.__name__)
                raise

            if not isinstance(out, dict):
                _LOGGER.warning("%s returned invalid result format: %s", stage.__class__.__name__, out)
                continue

            status, value = out.get("status"), out.get("result")
            if status == "handled":
                return value or None
            if status == "escalate":
                current = value
                continue
            if status == "error":
                return value or None

        _LOGGER.warning("All stages exhausted without a ConversationResult.")
        return None
