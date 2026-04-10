"""Shared fixtures for scenario integration tests.

Provides common mocking infrastructure used by all scenarios_*.py files.
Import these fixtures instead of duplicating them in each scenario file.
"""

import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation
from homeassistant.helpers import intent

from multistage_assist.conversation import MultiStageAssistAgent
from multistage_assist.stage0 import Stage0Processor
from multistage_assist.stage1_cache import Stage1CacheProcessor
from multistage_assist.stage2_llm import Stage2LLMProcessor
from multistage_assist.stage3_cloud import Stage3CloudProcessor
from multistage_assist import conversation_utils


class MockIntentResponse:
    """Mock for HA IntentResponse."""
    def __init__(self, language="en"):
        self.speech = {}
        self.response_type = None

    def async_set_speech(self, text, type="plain", extra_data=None):
        self.speech[type] = {"speech": text, "extra_data": extra_data}


@pytest.fixture(autouse=True)
def patch_intent_response():
    with patch(
        "multistage_assist.conversation_utils.intent.IntentResponse", MockIntentResponse
    ):
        yield


@pytest.fixture(autouse=True)
def patch_make_response():
    async def _mock_make_response(message, user_input, end=False):
        resp = MockIntentResponse(language=user_input.language or "de")
        resp.async_set_speech(message)
        res = MagicMock()
        res.response = resp
        res.conversation_id = user_input.conversation_id
        res.continue_conversation = not end
        return res

    p1 = patch(
        "multistage_assist.conversation.make_response", side_effect=_mock_make_response
    )

    if "custom_components.multistage_assist.conversation_utils" in sys.modules:
        mock_utils = sys.modules[
            "custom_components.multistage_assist.conversation_utils"
        ]
        mock_utils.make_response.side_effect = _mock_make_response

    with p1:
        yield


def create_base_agent(hass, config_entry, integration_llm_config):
    """Create a base agent with all stages wired up.
    
    Returns the agent instance. Caller should mock stage2.process as needed.
    """
    llm_config = integration_llm_config
    config_entry.data["stage1_ip"] = llm_config["stage1_ip"]
    config_entry.data["stage1_port"] = llm_config["stage1_port"]
    config_entry.data["stage1_model"] = llm_config["stage1_model"]

    agent = MultiStageAssistAgent(hass, config_entry.data)
    agent.stages = [
        Stage0Processor(hass, config_entry.data),
        Stage1CacheProcessor(hass, config_entry.data),
        Stage2LLMProcessor(hass, config_entry.data),
        Stage3CloudProcessor(hass, config_entry.data),
    ]
    for stage in agent.stages:
        stage.agent = agent

    return agent
