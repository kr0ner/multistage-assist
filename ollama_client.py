from __future__ import annotations

import logging
import aiohttp
import asyncio
import json

_LOGGER = logging.getLogger(__name__)


class OllamaClient:
    """Thin client for Ollama REST API."""

    _sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        import asyncio
        loop = asyncio.get_running_loop()
        if loop not in cls._sessions or cls._sessions[loop].closed:
            cls._sessions[loop] = aiohttp.ClientSession()
        return cls._sessions[loop]

    async def test_connection(self) -> bool:
        url = f"{self.base_url}/api/version"
        session = await self.get_session()
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            return True

    async def get_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        session = await self.get_session()
        async with session.get(url, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [m["name"] for m in data.get("models", [])]

    async def chat(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.25,
        num_ctx: int = 800,
        format: dict | None = None,
    ) -> str:
        """Send a chat request to Ollama."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "keep_alive": -1,  # Keep model loaded in memory permanently
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        }
        if format:
            payload["format"] = format

        _LOGGER.debug("Querying Ollama at %s (model: %s)", url, model)

        session = await self.get_session()
        async with session.post(url, json=payload, timeout=120) as resp:
            if resp.status == 400:
                err_text = await resp.text()
                _LOGGER.error("Ollama 400 Bad Request (Invalid Schema?): %s", err_text)
                resp.raise_for_status()
            resp.raise_for_status()
            data = await resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]

        _LOGGER.warning("Unexpected Ollama response: %s", data)
        return ""

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.25,
        num_ctx: int = 1024,
        format: dict | None = None,
    ) -> str:
        """Send a chat request with full message history."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        }
        if format:
            payload["format"] = format

        _LOGGER.debug(
            "Querying Ollama at %s (model: %s, messages: %d)",
            url, model, len(messages)
        )

        session = await self.get_session()
        async with session.post(url, json=payload, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]
            
        return ""
