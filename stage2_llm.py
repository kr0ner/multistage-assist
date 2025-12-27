"""Stage 2: LLM-based intent resolution.

Stage2 uses the local LLM (via keyword_intent capability) to derive
intent and entities when the semantic cache doesn't have a match.

Flow:
1. Use KeywordIntentCapability to parse intent from utterance
2. Use EntityResolverCapability to resolve entities
3. Detect chat intent → return escalate_chat for Stage3
4. Return StageResult.success if resolved, otherwise escalate to Stage3
"""

import logging
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation

from .base_stage import BaseStage
from .capabilities.keyword_intent import KeywordIntentCapability
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.area_resolver import AreaResolverCapability
from .capabilities.memory import MemoryCapability
from .capabilities.clarification import ClarificationCapability
from .capabilities.multi_turn_base import MultiTurnCapability
from .capabilities.timer import TimerCapability
from .capabilities.calendar import CalendarCapability
from .stage_result import StageResult
from .conversation_utils import with_new_text
from .constants.messages_de import SYSTEM_MESSAGES

_LOGGER = logging.getLogger(__name__)


# Chat detection patterns (user wants to chat, not control devices)
CHAT_PATTERNS = [
    r"\berzähl\b",
    r"\bwitz\b",
    r"\bjoke\b",
    r"\bstory\b",
    r"\bgeschichte\b",
    r"\bwer bist du\b",
    r"\bwas kannst du\b",
    r"\bhilfe\b",
    r"\bhelp\b",
]


class Stage2LLMProcessor(BaseStage):
    """Stage 2: LLM-based intent parsing for commands not in cache."""

    name = "stage2_llm"
    capabilities = [
        KeywordIntentCapability,
        EntityResolverCapability,
        AreaResolverCapability,
        MemoryCapability,
        ClarificationCapability,
        MultiTurnCapability,
        TimerCapability,
        CalendarCapability,
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}
        
        # Inject shared memory into capabilities that need it
        memory = self.get("memory")
        if self.has("entity_resolver"):
            self.get("entity_resolver").set_memory(memory)

    def _is_chat_request(self, text: str) -> bool:
        """Detect if user wants to chat rather than control devices."""
        import re
        text_lower = text.lower()
        for pattern in CHAT_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    async def _resolve_area_alias(self, area_name: str) -> Dict[str, Any]:
        """Resolve area alias using memory or LLM.
        
        Returns:
            Dict with:
            - match: resolved area name or None
            - unknown_area: original text if unresolved (optional)
            - candidates: list of available areas (optional)
        """
        if not area_name:
            return {"match": None}
            
        # Check memory first
        memory = self.get("memory")
        resolved = await memory.get_area_alias(area_name.lower())
        if resolved:
            return {"match": resolved}
            
        # Use area_resolver capability for fuzzy + LLM-based resolution
        if self.has("area_resolver"):
            resolver = self.get("area_resolver")
            result = await resolver.run(None, area_name=area_name)
            # Returns: {match, unknown_area, candidates}
            return result
                
        return {"match": area_name}

    async def process(
        self,
        user_input: conversation.ConversationInput,
        context: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        """Process user input using LLM-based intent parsing.
        
        Args:
            user_input: ConversationInput from Home Assistant
            context: Optional context from previous stages
            
        Returns:
            StageResult with status indicating outcome
        """
        context = context or {}
        
        _LOGGER.debug("[Stage2LLM] Input='%s'", user_input.text)

        # 0. Check for chat intent
        if self._is_chat_request(user_input.text):
            _LOGGER.debug("[Stage2LLM] Chat request detected → escalate_chat")
            return StageResult.escalate_chat(
                context={**context, "chat_detected": True},
                raw_text=user_input.text,
            )

        # 1. Get clarified commands from Stage1 or run clarification ourselves
        clarified_commands = context.get("commands", [])
        
        if not clarified_commands:
            # Stage1 didn't provide clarified commands - run clarification ourselves
            clarification = self.get("clarification")
            if clarification:
                clarified_commands = await clarification.run(user_input) or []
        
        # 2. Handle based on clarification result
        if not clarified_commands:
            # Empty result → escalate to Stage3
            _LOGGER.debug("[Stage2LLM] Clarification returned empty → escalate")
            return StageResult.escalate(
                context={**context, "clarification_empty": True},
                raw_text=user_input.text,
            )
        
        if len(clarified_commands) == 1:
            # Single command - process directly
            clarified_text = clarified_commands[0]
            if clarified_text != user_input.text:
                _LOGGER.debug("[Stage2LLM] Using clarified: '%s' → '%s'", 
                             user_input.text, clarified_text)
                user_input = with_new_text(user_input, clarified_text)
            # Continue to single-command processing below
        else:
            # Multiple atomic commands - return multi_command for conversation.py to iterate
            _LOGGER.debug("[Stage2LLM] Multi-command (%d) → returning multi_command", len(clarified_commands))
            return StageResult.multi_command(
                commands=clarified_commands,
                context={**context},
                raw_text=user_input.text,
            )


        # 2. Use keyword_intent to parse intent
        ki_data = await self.use("keyword_intent", user_input) or {}
        intent_name = ki_data.get("intent")
        slots = ki_data.get("slots") or {}
        domain = ki_data.get("domain")

        if not intent_name:
            _LOGGER.debug("[Stage2LLM] No intent derived → escalate")
            return StageResult.escalate(
                context={**context, "llm_failed": True},
                raw_text=user_input.text,
            )

        _LOGGER.debug("[Stage2LLM] Intent='%s', domain='%s', slots=%s", 
                     intent_name, domain, list(slots.keys()))

        # 3. Resolve area aliases if present
        if slots.get("area"):
            area_result = await self._resolve_area_alias(slots["area"])
            resolved_area = area_result.get("match")
            
            if resolved_area:
                if resolved_area != slots["area"]:
                    _LOGGER.debug("[Stage2LLM] Area alias: '%s' → '%s'", 
                                slots["area"], resolved_area)
                slots["area"] = resolved_area
            elif area_result.get("unknown_area"):
                # Unknown area - ask user to clarify
                unknown = area_result["unknown_area"]
                candidates = area_result.get("candidates", [])
                
                area_list = ", ".join(candidates[:10])
                if len(candidates) > 10:
                    area_list += f" und {len(candidates) - 10} weitere"
                
                message = f"{SYSTEM_MESSAGES['unknown_area_ask'].format(alias=unknown)} ({area_list})"
                
                _LOGGER.info("[Stage2LLM] Unknown area '%s', asking user", unknown)
                
                return StageResult.pending(
                    pending_type="area_learning",
                    message=message,
                    pending_data={
                        "unknown_alias": unknown,
                        "candidates": candidates,
                        "original_slots": slots,
                        "intent": intent_name,
                        "domain": domain,
                        "original_text": user_input.text,
                    },
                    raw_text=user_input.text,
                )

        # 4. Entity resolution
        resolver = self.get("entity_resolver")
        entities_for_resolver = {**slots, "intent": intent_name}
        if domain:
            entities_for_resolver["domain"] = domain
            
        resolved = await resolver.run(user_input, entities=entities_for_resolver)
        resolved_ids = (resolved or {}).get("resolved_ids", [])
        filtered_not_exposed = (resolved or {}).get("filtered_not_exposed", [])

        _LOGGER.debug("[Stage2LLM] Resolved %d entities: %s", 
                     len(resolved_ids), resolved_ids)

        if not resolved_ids:
            # Intent resolved but no matching entities found
            # Return success with empty entities - ExecutionPipeline handles error response
            _LOGGER.debug("[Stage2LLM] No entities resolved → success (ExecutionPipeline handles)")
            execution_params = {k: v for k, v in slots.items() 
                               if k not in {"area", "room", "floor", "name", "entity", 
                                           "device", "label", "domain", "device_class", "entity_id"}}
            return StageResult.success(
                intent=intent_name,
                entity_ids=[],  # Empty - ExecutionPipeline will handle
                params={
                    **execution_params,
                    "requested_area": slots.get("area"),
                    "requested_device_class": slots.get("device_class"),
                },
                context={
                    **context,
                    "domain": domain,
                    "from_llm": True,
                    "no_entities_found": True,
                    "filtered_not_exposed": filtered_not_exposed,  # For helpful error message
                },
                raw_text=user_input.text,
            )


        # 5. Build execution params (exclude resolution-only keys)
        resolution_keys = {"area", "room", "floor", "name", "entity", 
                          "device", "label", "domain", "device_class", "entity_id"}
        execution_params = {k: v for k, v in slots.items() if k not in resolution_keys}

        # 6. Success - ready for execution
        return StageResult.success(
            intent=intent_name,
            entity_ids=resolved_ids,
            params=execution_params,
            context={
                **context,
                "domain": domain,
                "from_llm": True,
            },
            raw_text=user_input.text,
        )

    async def _process_multi_command(
        self,
        original_input,
        commands: List[str],
        context: Dict[str, Any]
    ) -> StageResult:
        """Process multiple atomic commands through the intent pipeline.
        
        Args:
            original_input: Original ConversationInput
            commands: List of atomic command strings
            context: Processing context
            
        Returns:
            Merged StageResult with aggregated entity_ids
        """
        all_entity_ids: List[str] = []
        all_intents: List[str] = []
        merged_params: Dict[str, Any] = {}
        
        for cmd in commands:
            _LOGGER.debug("[Stage2LLM] Processing atomic command: '%s'", cmd)
            
            # Create modified input with atomic command
            cmd_input = with_new_text(original_input, cmd)
            
            # Process through keyword_intent
            ki_data = await self.use("keyword_intent", cmd_input) or {}
            intent_name = ki_data.get("intent")
            slots = ki_data.get("slots") or {}
            domain = ki_data.get("domain")
            
            if not intent_name:
                _LOGGER.warning("[Stage2LLM] No intent for: '%s'", cmd)
                continue
            
            all_intents.append(intent_name)
            
            # Resolve area aliases
            if slots.get("area"):
                area_result = await self._resolve_area_alias(slots["area"])
                resolved_area = area_result.get("match")
                if resolved_area and resolved_area != slots["area"]:
                    slots["area"] = resolved_area
            
            # Entity resolution
            resolver = self.get("entity_resolver")
            entities_for_resolver = {**slots, "intent": intent_name}
            if domain:
                entities_for_resolver["domain"] = domain
                
            resolved = await resolver.run(cmd_input, entities=entities_for_resolver)
            resolved_ids = (resolved or {}).get("resolved_ids", [])
            
            if resolved_ids:
                all_entity_ids.extend(resolved_ids)
            
            # Merge params (exclude resolution keys)
            resolution_keys = {"area", "room", "floor", "name", "entity", 
                              "device", "label", "domain", "device_class", "entity_id"}
            for k, v in slots.items():
                if k not in resolution_keys:
                    merged_params[k] = v
        
        if not all_entity_ids:
            _LOGGER.warning("[Stage2LLM] Multi-command resolved no entities → escalate")
            return StageResult.escalate(
                context={**context, "multi_command_no_entities": True},
                raw_text=original_input.text,
            )
        
        # Use the first intent (they should all be the same for multi-area commands)
        primary_intent = all_intents[0] if all_intents else None
        
        _LOGGER.info("[Stage2LLM] Multi-command resolved %d entities with intent '%s'",
                    len(all_entity_ids), primary_intent)
        
        return StageResult.success(
            intent=primary_intent,
            entity_ids=all_entity_ids,
            params=merged_params,
            context={
                **context,
                "from_llm": True,
                "multi_command": True,
                "command_count": len(commands),
            },
            raw_text=original_input.text,
        )

    # ----------------------------------------------------------------
    # Multi-turn handling (disambiguation, follow-ups)
    # ----------------------------------------------------------------
    
    def store_pending(self, session_id: str, data: Dict[str, Any]):
        """Store pending state for multi-turn conversations."""
        self._pending[session_id] = data
        _LOGGER.debug("[Stage2LLM] Stored pending state for %s", session_id)

    def get_pending(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pending state for a session."""
        return self._pending.get(session_id)

    def clear_pending(self, session_id: str):
        """Clear pending state after resolution."""
        if session_id in self._pending:
            del self._pending[session_id]
            _LOGGER.debug("[Stage2LLM] Cleared pending state for %s", session_id)

    async def continue_pending(
        self,
        user_input,
        pending_data: Dict[str, Any],
    ) -> StageResult:
        """Continue multi-turn interaction started by this stage.
        
        Args:
            user_input: User's follow-up response
            pending_data: Stored context from when pending was created
            
        Returns:
            StageResult - success to execute, pending to re-ask, or error
        """
        pending_type = pending_data.get("type", "")
        
        if pending_type == "area_learning":
            return await self._continue_area_learning(user_input, pending_data)
        
        # Unknown pending type - clear and escalate
        _LOGGER.warning("[Stage2LLM] Unknown pending type: %s", pending_type)
        return StageResult.escalate(
            context={"unknown_pending_type": pending_type},
            raw_text=user_input.text,
        )

    async def _continue_area_learning(
        self,
        user_input,
        pending_data: Dict[str, Any],
    ) -> StageResult:
        """Handle user's response to unknown area question.
        
        Matches response to available areas and learns the alias.
        Then re-runs original command through pipeline.
        """
        user_response = user_input.text.strip().lower()
        unknown_alias = pending_data.get("unknown_alias", "")
        candidates = pending_data.get("candidates", [])
        original_text = pending_data.get("original_text", "")
        
        # Try to match user response to an area
        matched_area = None
        
        # 1. Check for exact match first
        for area in candidates:
            if area.lower() == user_response:
                matched_area = area
                break
                
        # 2. If no exact match, check for substring
        if not matched_area:
            for area in candidates:
                if area.lower() in user_response:
                    matched_area = area
                    break
        
        if not matched_area:
            # Re-prompt if no match
            area_list = ", ".join(candidates[:10])
            message = f"{SYSTEM_MESSAGES['unknown_area_not_matched'].format(alias=unknown_alias)} ({area_list})"
            
            _LOGGER.debug("[Stage2LLM] Area learning: no match for '%s'", user_response)
            
            return StageResult.pending(
                pending_type="area_learning",
                message=message,
                pending_data=pending_data,  # Keep same data
                raw_text=user_input.text,
            )
        
        # Learn the alias
        resolver = self.get("area_resolver")
        await resolver.learn_area_alias(unknown_alias, matched_area)
        
        _LOGGER.info("[Stage2LLM] Learned area alias: '%s' → '%s'", unknown_alias, matched_area)
        
        # Return success with context for re-running original command
        # The original_text will be processed through pipeline again
        return StageResult.success(
            intent=pending_data.get("intent", ""),
            entity_ids=[],  # Will be resolved when original command re-runs
            params={
                "learned_alias": unknown_alias,
                "learned_area": matched_area,
                "original_text": original_text,
                "rerun_command": True,  # Signal to conversation.py to re-run
            },
            context={
                "area_learned": True,
                "from_stage2": True,
            },
            raw_text=original_text,
        )
