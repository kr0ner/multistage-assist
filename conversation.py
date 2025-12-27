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
import time
from typing import Any, Dict, List, Optional

# Conversation timeout constants
PENDING_TIMEOUT_SECONDS = 15  # Ask again after 15 seconds
PENDING_MAX_RETRIES = 2  # Give up after 2 retries (total 30 seconds)

from homeassistant.components import conversation

from .stage0 import Stage0Processor
from .stage1_cache import Stage1CacheProcessor
from .stage2_llm import Stage2LLMProcessor
from .stage3_gemini import Stage3GeminiProcessor
from .stage_result import StageResult
from .execution_pipeline import ExecutionPipeline
from .conversation_utils import with_new_text

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
        
        # Store pending execution context by conversation_id
        # When ExecutionPipeline owns the conversation (disambiguation, slot-filling, etc.)
        # conversation.py just checks ownership, not the reason
        self._execution_pending: Dict[str, Dict[str, Any]] = {}
        
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

    def _cleanup_stale_pending(self, current_conv_id: str) -> None:
        """Remove stale pending states from OTHER conversations."""
        now = time.time()
        stale_ids = []
        
        for conv_id, pending_data in self._execution_pending.items():
            if conv_id == current_conv_id:
                continue  # Don't clean current conversation, handle separately
            
            created_at = pending_data.get("_created_at", 0)
            retry_count = pending_data.get("_retry_count", 0)
            age = now - created_at
            
            # Clean up if too old (>30 seconds total with retries)
            max_age = PENDING_TIMEOUT_SECONDS * (PENDING_MAX_RETRIES + 1)
            if age > max_age:
                stale_ids.append(conv_id)
                _LOGGER.info("[Pipeline] Cleaning up stale pending for conversation %s (age=%.1fs)", conv_id, age)
        
        for conv_id in stale_ids:
            del self._execution_pending[conv_id]

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        _LOGGER.info("Received utterance: %s", user_input.text)
        
        conv_id = user_input.conversation_id or "default"
        
        # CLEANUP: Remove any stale pending states from OTHER conversations
        self._cleanup_stale_pending(current_conv_id=conv_id)

        # FIRST: Check if ExecutionPipeline owns this conversation
        # (could be disambiguation, slot-filling, follow-up - we don't care why)
        if conv_id in self._execution_pending:
            pending_data = self._execution_pending.pop(conv_id)
            remaining_commands = pending_data.pop("remaining_multi_commands", None)
            created_at = pending_data.pop("_created_at", None)
            retry_count = pending_data.pop("_retry_count", 0)
            
            # Check if this pending is stale (user took too long to respond)
            if created_at:
                age = time.time() - created_at
                if age > PENDING_TIMEOUT_SECONDS:
                    if retry_count >= PENDING_MAX_RETRIES:
                        # Too many retries - give up and start fresh
                        _LOGGER.info("[Pipeline] Pending timeout after %d retries, clearing state", retry_count)
                        # Fall through to process new command fresh
                    else:
                        # Re-ask the question
                        _LOGGER.info("[Pipeline] Pending timeout (%.1fs), re-asking (retry %d)", age, retry_count + 1)
                        pending_data["_created_at"] = time.time()
                        pending_data["_retry_count"] = retry_count + 1
                        if remaining_commands:
                            pending_data["remaining_multi_commands"] = remaining_commands
                        self._execution_pending[conv_id] = pending_data
                        # Re-create the disambiguation question
                        exec_result = await self._execution_pipeline.re_prompt_pending(
                            user_input, pending_data
                        )
                        if exec_result and exec_result.response:
                            return exec_result.response
                        # If re-prompt fails, continue with user's new input
            
            _LOGGER.debug("[Pipeline] ExecutionPipeline owns conversation %s, continuing", conv_id)
            
            exec_result = await self._execution_pipeline.continue_pending(
                user_input, pending_data
            )
            
            # If still pending, re-store (with remaining commands preserved and fresh timestamp)
            if exec_result.pending_data:
                exec_result.pending_data["_created_at"] = time.time()
                exec_result.pending_data["_retry_count"] = 0
                if remaining_commands:
                    exec_result.pending_data["remaining_multi_commands"] = remaining_commands
                self._execution_pending[conv_id] = exec_result.pending_data
                if exec_result.response:
                    return exec_result.response
                return await self._fallback(user_input)

            
            # Current command completed - now process remaining commands if any
            responses = []
            if exec_result.response:
                responses.append(exec_result.response)
            
            if remaining_commands:
                _LOGGER.info("[Pipeline] Resuming %d remaining multi-commands", len(remaining_commands))
                for i, cmd in enumerate(remaining_commands):
                    _LOGGER.debug("[Pipeline] Remaining command %d/%d: '%s'", i + 1, len(remaining_commands), cmd)
                    cmd_input = with_new_text(user_input, cmd)
                    cmd_response = await self._run_pipeline(cmd_input, context={})
                    
                    if not cmd_response:
                        continue
                    
                    # Check if this remaining command also triggered pending
                    if conv_id in self._execution_pending:
                        # Store the rest for later
                        rest = remaining_commands[i + 1:]
                        if rest:
                            self._execution_pending[conv_id]["remaining_multi_commands"] = rest
                            _LOGGER.info("[Pipeline] Remaining command paused, %d more remaining", len(rest))
                        return cmd_response
                    
                    responses.append(cmd_response)
            
            # Combine all responses
            if responses:
                speeches = []
                for resp in responses:
                    if hasattr(resp, 'response') and hasattr(resp.response, 'speech'):
                        speech = resp.response.speech.get("plain", {}).get("speech", "")
                        if speech:
                            speeches.append(speech)
                
                if speeches and len(speeches) > 1:
                    combined = " ".join(speeches)
                    first_resp = responses[0]
                    if hasattr(first_resp, 'response'):
                        first_resp.response.async_set_speech(combined)
                    return first_resp
                return responses[0]
            
            return await self._fallback(user_input)


        # SECOND: If any stage owns a pending turn, let it resolve first.
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

        # THIRD: Run stages pipeline
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
                    
                    # Store pending data if ExecutionPipeline needs to continue
                    if exec_result.pending_data:
                        conv_id = user_input.conversation_id or "default"
                        exec_result.pending_data["_created_at"] = time.time()
                        exec_result.pending_data["_retry_count"] = 0
                        self._execution_pending[conv_id] = exec_result.pending_data
                        _LOGGER.debug("[Pipeline] ExecutionPipeline taking ownership of %s", conv_id)
                    
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
        
            elif result.status == "multi_command":
                # Process each atomic command through full pipeline independently
                # CRITICAL: Only proceed to next command when current fully completes
                _LOGGER.info("[Pipeline] Processing %d atomic commands", len(result.commands))
                all_responses = []
                
                for i, cmd in enumerate(result.commands):
                    _LOGGER.debug("[Pipeline] Multi-command %d/%d: '%s'", i + 1, len(result.commands), cmd)
                    # Create new input with this command text
                    cmd_input = with_new_text(user_input, cmd)
                    # Run through full pipeline with FRESH context (no contamination!)
                    cmd_response = await self._run_pipeline(cmd_input, context={})
                    
                    if not cmd_response:
                        _LOGGER.warning("[Pipeline] Multi-command %d/%d returned None", i + 1, len(result.commands))
                        continue
                    
                    # Check if this command triggered a pending state (disambiguation, slot-filling, etc.)
                    conv_id = user_input.conversation_id or "default"
                    if conv_id in self._execution_pending:
                        # Command is waiting for user response - stop here
                        remaining = result.commands[i + 1:]
                        if remaining:
                            # Store remaining commands to process after pending resolves
                            self._execution_pending[conv_id]["remaining_multi_commands"] = remaining
                        # Ensure timestamp is set
                        if "_created_at" not in self._execution_pending[conv_id]:
                            self._execution_pending[conv_id]["_created_at"] = time.time()
                            self._execution_pending[conv_id]["_retry_count"] = 0
                        _LOGGER.info(
                                "[Pipeline] Multi-command paused at %d/%d, %d commands remaining",
                                i + 1, len(result.commands), len(remaining)
                            )
                        # Return the response asking user for clarification
                        return cmd_response
                    
                    # Command completed successfully or with error - collect response
                    all_responses.append(cmd_response)
                
                # All commands completed without pending
                if all_responses:
                    # Get speech from each response and combine
                    speeches = []
                    for resp in all_responses:
                        if hasattr(resp, 'response') and hasattr(resp.response, 'speech'):
                            speech = resp.response.speech.get("plain", {}).get("speech", "")
                            if speech:
                                speeches.append(speech)
                    
                    if speeches:
                        combined = " ".join(speeches)
                        # Return first response with combined speech
                        first_resp = all_responses[0]
                        if hasattr(first_resp, 'response'):
                            first_resp.response.async_set_speech(combined)
                        return first_resp
                    return all_responses[0]
                # No responses - fall through to end

                
        
            elif result.status == "error":
                if result.response:
                    return result.response
                continue

        _LOGGER.warning("All stages exhausted without result")
        return None
