"""Stage 3: Gemini Cloud-based intent resolution and chat.

Stage3 is the final fallback that uses Google Gemini for:
1. Intent derivation when local LLM fails
2. General chat/conversation (jokes, help, etc.)

The stage checks `context["chat_mode"]` to skip full context when
the user just wants to chat, saving API costs.

Flow:
1. If chat_mode: Direct chat response (minimal context)
2. Otherwise: Build full system prompt with domain configs + entities
3. Single API call for intent derivation OR chat response
"""

import logging
import json
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation

from .base_stage import BaseStage
from .capabilities.chat import ChatCapability
from .capabilities.google_gemini_client import GoogleGeminiClient, GeminiError
from .stage_result import StageResult
from .conversation_utils import make_response
from .constants.messages_de import ERROR_MESSAGES
from .const import CONF_STAGE3_MODEL

_LOGGER = logging.getLogger(__name__)


# System prompt for intent derivation
INTENT_SYSTEM_PROMPT = """You are a Smart Home Assistant.

Task: Analyze the user input and extract the intent.

Available Intents:
- HassTurnOn: Turn on (Light on, open blinds)
- HassTurnOff: Turn off (Light off, close blinds)
- HassLightSet: Set brightness/color (Dim light, to 50%)
- HassSetPosition: Set position (Blinds to 50%)
- HassGetState: Get state (Is the light on?)
- HassClimateSetTemperature: Set temperature (Heater to 21 degrees)
- TemporaryControl: Time-limited (on for 10 minutes)
- DelayedControl: Delayed (off in 10 minutes)
- HassTimerSet: Set timer

If the user asks a general question or wants to chat, answer with:
{{"mode": "chat", "response": "Your answer here in German"}}

For a Smart Home command, answer with:
{{"mode": "intent", "intent": "IntentName", "area": "Area", "domain": "light/cover/switch/climate", "params": {{}}}}

Available Areas: {areas}
Available Floors: {floors}

User Input: {user_input}
IMPORTANT: The user speaks German. You must output the 'response' field in German.
"""

# Minimal chat prompt (when chat_mode is True)
CHAT_SYSTEM_PROMPT = """You are a friendly Smart Home Assistant.
Answer briefly and naturally in German (Du-form).
The user wants to chat, not control devices."""


class Stage3GeminiProcessor(BaseStage):
    """Stage 3: Gemini cloud for final fallback and chat."""

    name = "stage3_gemini"
    capabilities = [ChatCapability]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._chat_sessions: Dict[str, List[Dict[str, str]]] = {}
        
        # Initialize Gemini client if API key is configured
        api_key = config.get("gemini_api_key") or config.get("google_api_key")
        self._gemini_client = None
        if api_key:
            model = config.get(CONF_STAGE3_MODEL, "gemini-2.5-flash")
            self._gemini_client = GoogleGeminiClient(api_key, model)
            _LOGGER.info("[Stage3Gemini] Initialized with model: %s", model)
        else:
            _LOGGER.warning("[Stage3Gemini] No API key configured")

    def _get_areas_and_floors(self) -> tuple[List[str], List[str]]:
        """Get available areas and floors from Home Assistant."""
        areas = []
        floors = []
        
        try:
            from homeassistant.helpers import area_registry as ar, floor_registry as fr
            
            area_registry = ar.async_get(self.hass)
            for area in area_registry.async_list_areas():
                areas.append(area.name)
                
            floor_registry = fr.async_get(self.hass)
            for floor in floor_registry.async_list_floors():
                floors.append(floor.name)
        except Exception as e:
            _LOGGER.debug("[Stage3Gemini] Could not get areas/floors: %s", e)
            
        return areas, floors

    def _get_session_key(self, user_input) -> str:
        """Get session key for chat history."""
        return getattr(user_input, "session_id", None) or user_input.conversation_id

    async def process(
        self,
        user_input: conversation.ConversationInput,
        context: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        """Process user input using Gemini cloud.
        
        Args:
            user_input: ConversationInput from Home Assistant
            context: Context from previous stages (may include chat_mode=True)
            
        Returns:
            StageResult with status indicating outcome
        """
        context = context or {}
        
        _LOGGER.debug("[Stage3Gemini] Input='%s', chat_mode=%s", 
                     user_input.text, context.get("chat_mode"))

        if not self._gemini_client:
            _LOGGER.error("[Stage3Gemini] No Gemini client available")
            return StageResult.error(
                response=await make_response(
                    ERROR_MESSAGES["gemini_unavailable"],
                    user_input
                ),
                raw_text=user_input.text,
            )

        session_key = self._get_session_key(user_input)

        # Chat mode: Direct chat response with minimal context
        if context.get("chat_mode"):
            return await self._handle_chat(user_input, session_key, context)

        # Intent mode: Try to derive intent with full context
        return await self._handle_intent(user_input, session_key, context)

    async def _handle_chat(
        self,
        user_input,
        session_key: str,
        context: Dict[str, Any]
    ) -> StageResult:
        """Handle chat-only mode with minimal context."""
        _LOGGER.debug("[Stage3Gemini] Chat mode - minimal context")

        # Get or create chat history
        if session_key not in self._chat_sessions:
            self._chat_sessions[session_key] = []
        history = self._chat_sessions[session_key]

        # Build message with chat system prompt
        full_prompt = f"{CHAT_SYSTEM_PROMPT}\n\nBenutzer: {user_input.text}"

        try:
            response_text = await self._gemini_client.chat(full_prompt, history)
            
            # Update history
            history.append({"role": "user", "content": user_input.text})
            history.append({"role": "assistant", "content": response_text})

            return StageResult(
                status="success",
                intent=None,  # No intent - this is chat
                response=await make_response(response_text, user_input),
                context={**context, "chat_response": True},
                raw_text=user_input.text,
            )
        except GeminiError as e:
            _LOGGER.warning("[Stage3Gemini] Chat GeminiError: %s (type=%s)", e, e.error_type)
            error_msg = ERROR_MESSAGES.get(e.error_type, ERROR_MESSAGES["api_error"])
            return StageResult.error(
                response=await make_response(error_msg, user_input),
                raw_text=user_input.text,
            )
        except Exception as e:
            _LOGGER.exception("[Stage3Gemini] Chat error: %s", e)
            return StageResult.error(
                response=await make_response(
                    ERROR_MESSAGES["api_error"],
                    user_input
                ),
                raw_text=user_input.text,
            )

    async def _handle_intent(
        self,
        user_input,
        session_key: str,
        context: Dict[str, Any]
    ) -> StageResult:
        """Handle intent derivation with full context."""
        _LOGGER.debug("[Stage3Gemini] Intent mode - full context")

        # Get areas and floors for context
        areas, floors = self._get_areas_and_floors()

        from .constants.messages_de import SYSTEM_MESSAGES
        
        # Build prompt with full context
        prompt = INTENT_SYSTEM_PROMPT.format(
            areas=", ".join(areas) if areas else SYSTEM_MESSAGES["none_known"],
            floors=", ".join(floors) if floors else SYSTEM_MESSAGES["none_known"],
            user_input=user_input.text,
        )

        try:
            response_text = await self._gemini_client.chat(prompt, [])
            
            # Try to parse JSON response
            result = self._parse_gemini_response(response_text)
            
            if result.get("mode") == "chat":
                # Gemini decided this is a chat, not a command
                chat_response = result.get("response", response_text)
                return StageResult(
                    status="success",
                    intent=None,
                    response=await make_response(chat_response, user_input),
                    context={**context, "gemini_chat": True},
                    raw_text=user_input.text,
                )

            if result.get("mode") == "intent" and result.get("intent"):
                # Gemini derived an intent
                return StageResult.success(
                    intent=result["intent"],
                    entity_ids=[],  # Will be resolved by ExecutionPipeline
                    params={
                        "area": result.get("area"),
                        "floor": result.get("floor"),
                        "domain": result.get("domain"),
                        **(result.get("params") or {}),
                    },
                    context={**context, "from_gemini": True},
                    raw_text=user_input.text,
                )

            # Couldn't parse - treat as error
            _LOGGER.warning("[Stage3Gemini] Unexpected response format: %s", response_text)
            return StageResult.error(
                response=await make_response(
                    ERROR_MESSAGES["not_understood"],
                    user_input
                ),
                raw_text=user_input.text,
            )

        except GeminiError as e:
            _LOGGER.warning("[Stage3Gemini] Intent GeminiError: %s (type=%s)", e, e.error_type)
            error_msg = ERROR_MESSAGES.get(e.error_type, ERROR_MESSAGES["api_error"])
            return StageResult.error(
                response=await make_response(error_msg, user_input),
                raw_text=user_input.text,
            )
        except Exception as e:
            _LOGGER.exception("[Stage3Gemini] Intent derivation error: %s", e)
            return StageResult.error(
                response=await make_response(
                    ERROR_MESSAGES["api_error"],
                    user_input
                ),
                raw_text=user_input.text,
            )

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response."""
        try:
            # Try to find JSON in response
            text = response_text.strip()
            
            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            
            return json.loads(text)
        except json.JSONDecodeError:
            _LOGGER.debug("[Stage3Gemini] Could not parse JSON: %s", response_text[:100])
            return {"mode": "chat", "response": response_text}

    def clear_chat_session(self, user_input):
        """Clear chat history for a session."""
        key = self._get_session_key(user_input)
        if key in self._chat_sessions:
            del self._chat_sessions[key]
            _LOGGER.debug("[Stage3Gemini] Cleared chat session: %s", key)
