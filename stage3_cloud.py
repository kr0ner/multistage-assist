"""Stage 3: Cloud-based reasoning and conversation.

Unified Stage 3 handles multi-provider LLMs (Gemini, OpenAI, Anthropic, Grok).
It uses a multi-turn reasoning loop with MCP tools.
Chat is an emergent behavior of the LLM response.
"""

from __future__ import annotations
import logging
import json
import asyncio
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation

from .base_stage import BaseStage
from .capabilities.llm_providers import (
    LLMProvider,
    GeminiProvider,
    OpenAIProvider,
    AnthropicProvider,
)
from .capabilities.mcp import McpToolCapability
from .stage_result import StageResult
from .conversation_utils import make_response
from .constants.messages_de import ERROR_MESSAGES, SYSTEM_PROMPT_STAGE3
from .const import (
    CONF_STAGE3_PROVIDER,
    CONF_STAGE3_MODEL,
    CONF_GOOGLE_API_KEY,
    CONF_OPENAI_API_KEY,
    CONF_ANTHROPIC_API_KEY,
    CONF_GROK_API_KEY,
)

_LOGGER = logging.getLogger(__name__)


class Stage3CloudProcessor(BaseStage):
    """Stage 3: Cloud reasoning engine."""

    name = "stage3_cloud"
    capabilities = [McpToolCapability]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.provider_type = config.get(CONF_STAGE3_PROVIDER, "gemini")
        self.model_name = config.get(CONF_STAGE3_MODEL)
        
        self.api_key = self._get_api_key()
        self.provider: Optional[LLMProvider] = self._init_provider()
        
        # Session history storage
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def _get_api_key(self) -> Optional[str]:
        if self.provider_type == "gemini":
            return self.config.get(CONF_GOOGLE_API_KEY)
        if self.provider_type in ["openai", "grok"]:
            return self.config.get(CONF_OPENAI_API_KEY) or self.config.get(CONF_GROK_API_KEY)
        if self.provider_type == "anthropic":
            return self.config.get(CONF_ANTHROPIC_API_KEY)
        return None

    def _init_provider(self) -> Optional[LLMProvider]:
        if not self.api_key:
            return None
            
        model = self.model_name
        if not model:
            model = {
                "gemini": "gemini-2.0-flash-lite",
                "openai": "gpt-4o-mini",
                "grok": "grok-2-1212",
                "anthropic": "claude-3-5-sonnet-latest"
            }.get(self.provider_type)

        if self.provider_type == "gemini":
            return GeminiProvider(self.api_key, model)
        if self.provider_type == "openai":
            return OpenAIProvider(self.api_key, model)
        if self.provider_type == "grok":
            return OpenAIProvider(self.api_key, model, base_url="https://api.x.ai/v1")
        if self.provider_type == "anthropic":
            return AnthropicProvider(self.api_key, model)
        return None

    def _get_session_key(self, user_input) -> str:
        return user_input.conversation_id or "default"

    async def process(
        self,
        user_input: conversation.ConversationInput,
        context: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        if not self.provider:
            return StageResult.error(
                response=await make_response(ERROR_MESSAGES["llm_not_configured"], user_input),
                raw_text=user_input.text
            )

        session_key = self._get_session_key(user_input)
        if session_key not in self._sessions:
            self._sessions[session_key] = []
        
        history = self._sessions[session_key]
        
        # Build messages for the provider
        messages = [{"role": "system", "content": SYSTEM_PROMPT_STAGE3}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input.text})

        # Get tools from MCP capability
        mcp = self.get("mcp_tool")
        tools = mcp.get_tools() if mcp else []

        max_turns = 5
        for turn in range(max_turns):
            _LOGGER.debug("[Stage 3] Reasoning turn %d", turn)
            
            # OpenAI/Grok support tools, Gemini in this version uses text-based reasoning for now
            # (can be expanded later to full SDK tool use)
            resp = await self.provider.chat(messages, tools=tools if turn < max_turns-1 else None)
            
            content = resp.get("content", "")
            tool_calls = resp.get("tool_calls", [])

            if not tool_calls:
                # Final answer
                history.append({"role": "user", "content": user_input.text})
                history.append({"role": "assistant", "content": content})
                
                # Keep history within limits (10 messages = 5 turns)
                if len(history) > 10:
                    self._sessions[session_key] = history[-10:]
                
                return StageResult(
                    status="success",
                    intent=None, # It's a straight chat/resp
                    response=await make_response(content, user_input),
                    context={**(context or {}), "from_stage3": True},
                    raw_text=user_input.text
                )

            # Process tool calls
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc in tool_calls:
                tool_name = tc["name"]
                args = tc["args"]
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {}
                
                _LOGGER.info("[Stage 3] Tool Call: %s(%s)", tool_name, args)
                result = await mcp.execute_tool(tool_name, args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": tool_name,
                    "content": json.dumps(result)
                })

        return StageResult.error(
            response=await make_response("Ich konnte keine Lösung finden.", user_input),
            raw_text=user_input.text
        )
