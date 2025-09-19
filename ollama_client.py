from __future__ import annotations

import enum
import logging
import aiohttp

from .const import (
    CONF_STAGE1_IP,
    CONF_STAGE1_PORT,
    CONF_STAGE1_MODEL,
    CONF_STAGE2_IP,
    CONF_STAGE2_PORT,
    CONF_STAGE2_MODEL,
)

_LOGGER = logging.getLogger(__name__)


class Stage(enum.Enum):
    """Enum for Multi-Stage Assist stages."""

    STAGE1 = 1
    STAGE2 = 2


def _get_stage_config(config: dict, stage: Stage) -> tuple[str, int, str]:
    """Return (ip, port, model) for the given stage."""
    if stage == Stage.STAGE1:
        return (
            config[CONF_STAGE1_IP],
            config[CONF_STAGE1_PORT],
            config[CONF_STAGE1_MODEL],
        )
    elif stage == Stage.STAGE2:
        return (
            config[CONF_STAGE2_IP],
            config[CONF_STAGE2_PORT],
            config[CONF_STAGE2_MODEL],
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


async def query_ollama(
    config: dict,
    stage: Stage,
    system_prompt: str,
    prompt: str,
) -> str:
    """Send a chat request to Ollama at given stage."""
    ip, port, model = _get_stage_config(config, stage)
    url = f"http://{ip}:{port}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    _LOGGER.debug("Querying Ollama %s at %s with model=%s", stage, url, model)

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()

    # Ollama returns either "message" or "response"
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "response" in data:
        return data["response"]

    _LOGGER.warning("Unexpected Ollama response: %s", data)
    return ""
