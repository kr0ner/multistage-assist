import abc
import logging
from typing import Any, Dict, List, Optional, Union

_LOGGER = logging.getLogger(__name__)

class LLMProvider(abc.ABC):
    """Abstract base class for cloud LLM providers."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abc.abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Send a list of messages to the LLM and get a response.
        Returns a dict with 'content' (str) and optionally 'tool_calls' (list).
        """
        pass

class GeminiProvider(LLMProvider):
    """Google Gemini Provider using google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        super().__init__(api_key, model)
        self._client = None
        self._types = None

    async def _ensure_client(self):
        if self._client:
            return
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=self.api_key)
        self._types = types

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        gemini_messages = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": m["content"]}]})
        return gemini_messages

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        await self._ensure_client()
        contents = self._format_messages(messages)
        
        # Simple implementation for now - ignoring tools for Gemini until SDK tool usage is verified
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                )
            )
            return {"content": response.text if response and response.text else ""}
        except Exception as e:
            _LOGGER.error("Gemini Provider Error: %s", e)
            return {"content": f"Fehler bei Gemini: {str(e)}"}

class OpenAIProvider(LLMProvider):
    """OpenAI/Grok Provider using openai SDK."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        super().__init__(api_key, model)
        self.base_url = base_url
        self._client = None

    async def _ensure_client(self):
        if self._client:
            return
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        await self._ensure_client()
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                # Map standard tools to OpenAI format if needed
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
                kwargs["tool_choice"] = "auto"
            
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            
            res = {"content": choice.message.content or ""}
            if choice.message.tool_calls:
                res["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": tc.function.arguments # JSON string in OpenAI
                    } for tc in choice.message.tool_calls
                ]
            return res
        except Exception as e:
            _LOGGER.error("OpenAI/Grok Provider Error: %s", e)
            return {"content": f"Fehler bei Provider: {str(e)}"}

class AnthropicProvider(LLMProvider):
    """Anthropic Claude Provider."""
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        # Placeholder for implementation
        return {"content": "Claude Provider nicht implementiert."}
