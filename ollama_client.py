from __future__ import annotations

import logging
import aiohttp

_LOGGER = logging.getLogger(__name__)


class OllamaClient:
    """Thin client for Ollama REST API."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"

    async def test_connection(self) -> bool:
        url = f"{self.base_url}/api/version"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return True

    async def get_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        async with aiohttp.ClientSession() as session:
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
            "options": {"num_ctx": 1024, "temperature": temperature},
        }

        _LOGGER.debug("Querying Ollama at %s with model=%s", url, model)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        if "response" in data:
            return data["response"]

        _LOGGER.warning("Unexpected Ollama response: %s", data)
        return ""
