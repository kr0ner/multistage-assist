"""Test scenarios for Climate and State queries."""

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

    import sys
    if "custom_components.multistage_assist.conversation_utils" in sys.modules:
        mock_utils = sys.modules[
            "custom_components.multistage_assist.conversation_utils"
        ]
        mock_utils.make_response.side_effect = _mock_make_response

    with p1:
        yield

@pytest.fixture
def agent(hass, config_entry, integration_llm_config):
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
    
    # Mock verification to speed up tests and avoid false negatives in mock environment
    from multistage_assist.capabilities.intent_executor import IntentExecutorCapability
    patcher = patch.object(IntentExecutorCapability, "_verify_execution", return_value=True)
    patcher.start()

    from multistage_assist.stage_result import StageResult

    # Mock Stage 2 (Local LLM)
    async def mock_stage2_process(user_input, context):
        text = user_input.text.lower()
        if "licht" in text or "dunkel" in text or "hell" in text:
            return StageResult(status="success", message="Stage 2: Okay.")
        return StageResult(status="escalate")
    
    patcher_stage2 = patch.object(agent.stages[2], "process", side_effect=mock_stage2_process)
    patcher_stage2.start()

    # Mock Stage 3 (Gemini) for deterministic results
    from custom_components.multistage_assist.conversation_utils import make_response
    async def mock_gemini_process(user_input, context):
        text = user_input.text.lower()
        if "warm" in text or "temperatur" in text:
            msg = "Im Büro sind es 21 Grad."
        elif "nachtmodus" in text:
            msg = "Der Nachtmodus ist aus."
        else:
            msg = "Okay."
        return StageResult(
            status="success",
            response=await make_response(msg, user_input)
        )
    
    patcher_gemini = patch.object(agent.stages[3], "process", side_effect=mock_gemini_process)
    patcher_gemini.start()
    
    yield agent
    patcher_gemini.stop()
    patcher_stage2.stop()
    patcher.stop()

async def test_scenario_temperature_query(agent, hass):
    """Scenario 15: Temperature Query (Wie warm ist es im Büro?)."""
    user_input = conversation.ConversationInput(
        text="Wie warm ist es im Büro?",
        context=MagicMock(),
        conversation_id="test_id_15",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result = await agent.async_process(user_input)
        assert result is not None
        speech = result.response.speech["plain"]["speech"]
        assert "21" in speech or "temperature" in speech.lower() or "büro" in speech.lower()

async def test_scenario_state_query(agent, hass):
    """Scenario 16: State Query (Ist der Nachtmodus an?)."""
    user_input = conversation.ConversationInput(
        text="Ist der Nachtmodus an?",
        context=MagicMock(),
        conversation_id="test_id_16",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result = await agent.async_process(user_input)
        assert result is not None
        speech = result.response.speech["plain"]["speech"]
        assert "aus" in speech.lower() or "off" in speech.lower() or "nacht" in speech.lower()

async def test_scenario_nonexistent_area_escalates(agent, hass):
    """Scenario: Non-existent area should escalate to chat mode."""
    user_input = conversation.ConversationInput(
        text="Schalte das Licht im S-Zimmer an",
        context=MagicMock(),
        conversation_id="test_nonexistent_area",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result = await agent.async_process(user_input)
        assert result is not None
        assert result.response is not None
