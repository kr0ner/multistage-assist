"""Stage 1: Semantic Cache-based intent resolution.

Stage1 uses the semantic cache to fast-track commands that have been
successfully executed before. This provides instant responses for
repeated or similar commands without LLM calls.

Flow:
1. Use ClarificationCapability to split/clean user input
2. For each command: lookup in SemanticCacheCapability
3. Bypass cache for custom intents (TemporaryControl, HassTimerSet)
4. Return StageResult.success if cache hit, otherwise escalate to Stage2
"""

import logging
import re
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation

from .base_stage import BaseStage
from .capabilities.clarification import ClarificationCapability
from .capabilities.semantic_cache import SemanticCacheCapability
from .capabilities.memory import MemoryCapability
from .stage_result import StageResult
from .conversation_utils import with_new_text
from .utils.german_utils import extract_delay, extract_duration

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# CACHE BYPASS INTENTS
# ============================================================================
# These intents should ALWAYS go to LLM, never be cached or matched from cache.
#
# WHY TIMER/CALENDAR BYPASS CACHE:
# Timer and calendar commands often have contextual information that the
# cache normalization strips out. For example:
#   "Stelle einen Timer auf 15 Minuten der mich an das Gulasch erinnert"
# The cache normalizes this to just "Timer auf Minuten" and loses the
# crucial "der mich an das Gulasch erinnert" part.
#
# The LLM can properly parse the full command including:
#   - Timer name/description
#   - Calendar event details
#   - Reminder context
#
# Simple device control commands (lights, covers, etc.) are safe to cache
# because they have predictable structure with extractable numeric values.
# ============================================================================
CACHE_BYPASS_INTENTS: set = {
    "HassTimerSet",      # Timer commands need full context (name, description)
    "HassTimerCancel",   # Timer cancel needs timer name
    # Future: Add calendar intents here when implemented
}


class Stage1CacheProcessor(BaseStage):
    """Stage 1: Semantic cache lookup for fast command execution."""

    name = "stage1_cache"
    capabilities = [
        ClarificationCapability,
        SemanticCacheCapability,
        MemoryCapability,
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        
        # Cache-only mode - if enabled and cache misses, we escalate directly
        self._cache_only_mode = config.get("skip_stage1_llm", False)
        if self._cache_only_mode:
            _LOGGER.info("[Stage1Cache] Running in cache-only mode")

    async def _normalize_area_aliases(self, user_input) -> conversation.ConversationInput:
        """Preprocess user input to normalize common area aliases using memory.
        
        E.g., "bad" → "Badezimmer", "ezi" → "Esszimmer"
        """
        text = user_input.text
        words = text.lower().split()

        memory_cap = self.get("memory")
        
        for word in words:
            clean_word = word.strip(".,!?")
            if not clean_word:
                continue

            normalized = await memory_cap.get_area_alias(clean_word)
            if normalized:
                _LOGGER.debug("[Stage1Cache] Alias: '%s' → '%s'", clean_word, normalized)
                import re
                pattern = re.compile(re.escape(clean_word), re.IGNORECASE)
                text = pattern.sub(normalized, text, count=1)

        if text != user_input.text:
            _LOGGER.debug("[Stage1Cache] Normalized: %s → %s", user_input.text, text)
            return with_new_text(user_input, text)
        return user_input

    async def process(
        self,
        user_input: conversation.ConversationInput,
        context: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        """Process user input using semantic cache lookup.
        
        Args:
            user_input: ConversationInput from Home Assistant
            context: Optional context from Stage0 (NLU data)
            
        Returns:
            StageResult with status indicating outcome
        """
        context = context or {}
        
        _LOGGER.debug("[Stage1Cache] Input='%s'", user_input.text)

        # 1. Normalize area aliases
        user_input = await self._normalize_area_aliases(user_input)

        # 2. Use clarification to split multi-commands
        clarification_cap = self.get("clarification")
        commands = await clarification_cap.run(user_input)
        
        if not commands:
            commands = [user_input.text]
        
        # For multi-commands, return multi_command status
        # conversation.py will iterate each command through full pipeline
        if len(commands) > 1:
            _LOGGER.debug("[Stage1Cache] Multi-command detected (%d) → multi_command", len(commands))
            return StageResult.multi_command(
                commands=commands,
                context={**context},
                raw_text=user_input.text,
            )


        # 3. Check if the command should bypass cache
        # (Context may already have NLU intent from Stage0)
        nlu_intent = context.get("nlu_intent", "")
        if nlu_intent in CACHE_BYPASS_INTENTS:
            _LOGGER.debug("[Stage1Cache] Bypass cache for intent '%s' → escalate", nlu_intent)
            return StageResult.escalate(
                context={**context, "cache_bypassed": True, "bypass_reason": "custom_intent"},
                raw_text=user_input.text,
            )


        # 4. Semantic cache lookup
        if not self.has("semantic_cache"):
            _LOGGER.debug("[Stage1Cache] No semantic cache configured → escalate")
            return StageResult.escalate(context=context, raw_text=user_input.text)

        cache = self.get("semantic_cache")
        cached = await cache.lookup(user_input.text)

        if not cached:
            _LOGGER.debug("[Stage1Cache] Cache MISS → escalate")
            # Include clarified command(s) so Stage2 can use them
            return StageResult.escalate(
                context={**context, "cache_miss": True, "commands": commands},
                raw_text=user_input.text,
            )

        # 5. Cache HIT
        _LOGGER.info(
            "[Stage1Cache] Cache HIT (%.3f): %s → %s",
            cached["score"], cached["intent"], cached["entity_ids"]
        )

        # NOTE: Disambiguation is handled by ExecutionPipeline, not here.
        # Stage1 just returns success with entities - pipeline handles the rest.

        # 6. Success! Ready for execution
        # Merge cache slots with NLU entities (NLU takes priority for state queries)
        cache_slots = {
            k: v for k, v in cached.get("slots", {}).items()
            if k not in ("name", "entity_id")
        }
        
        # Merge NLU entities into params (NLU freshly parsed the "aus"/"an" state)
        nlu_entities = context.get("nlu_entities", {})
        if nlu_entities:
            # State is particularly important - use NLU's interpretation
            if "state" in nlu_entities:
                cache_slots["state"] = nlu_entities["state"]
                _LOGGER.debug("[Stage1Cache] Using NLU state='%s' instead of cache", nlu_entities["state"])
        
        # For DelayedControl: Extract delay from raw text since cache uses generic patterns
        # "in 3 Minuten" / "um 15 Uhr" → delay slot
        if cached["intent"] == "DelayedControl":
            delay_str = extract_delay(user_input.text)
            if delay_str:
                cache_slots["delay"] = delay_str
                _LOGGER.debug("[Stage1Cache] Extracted delay='%s' from text", delay_str)
        
        # For TemporaryControl: Extract duration from raw text since cache uses generic patterns
        # "für 3 Minuten" → duration slot
        if cached["intent"] == "TemporaryControl":
            duration_str = extract_duration(user_input.text)
            if duration_str:
                cache_slots["duration"] = duration_str
                _LOGGER.debug("[Stage1Cache] Extracted duration='%s' from text", duration_str)
        
        # NOTE: HassTimerSet bypasses cache (see CACHE_BYPASS_INTENTS)
        # Timer commands need LLM to preserve context like "der mich an das Gulasch erinnert"

        return StageResult.success(
            intent=cached["intent"],
            entity_ids=cached["entity_ids"],
            params=cache_slots,
            context={
                **context,
                "from_cache": True,
                "cache_score": cached["score"],
            },
            raw_text=user_input.text,
        )


# Alias for backward compatibility during migration
Stage1Processor = Stage1CacheProcessor
