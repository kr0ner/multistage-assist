"""Test scenarios for Covers domain."""

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
        return StageResult(status="escalate")
    
    patcher_stage2 = patch.object(agent.stages[2], "process", side_effect=mock_stage2_process)
    patcher_stage2.start()

    # Mock Stage 3 (Gemini) for deterministic results
    from custom_components.multistage_assist.conversation_utils import make_response
    async def mock_gemini_process(user_input, context):
        return StageResult(
            status="success",
            response=await make_response("Okay.", user_input)
        )
    
    patcher_gemini = patch.object(agent.stages[3], "process", side_effect=mock_gemini_process)
    patcher_gemini.start()
    
    yield agent
    patcher_gemini.stop()
    patcher_stage2.stop()
    patcher.stop()

async def test_scenario_cover_plural_disambiguation(agent, hass):
    """Scenario 14: Cover Control with Plural -> Direct Execution."""
    user_input = conversation.ConversationInput(
        text="Fahre die Rolläden im Büro herunter",
        context=MagicMock(),
        conversation_id="test_id_14",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassTurnOff"
    mock_match.entities = {
        "area": MagicMock(value="Büro"),
        "domain": MagicMock(value="cover"),
    }

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        assert mock_async_handle.call_count == 2
        entity_ids = [call.kwargs["slots"]["name"]["value"] for call in mock_async_handle.call_args_list]
        assert set(entity_ids) == {"cover.buro_nord", "cover.buro_ost"}

async def test_timebox_cover(agent, hass):
    """Parametrized test for timebox functionality for covers."""
    entity_id = "cover.buro_nord"
    value = 75
    duration = "10 Minuten"
    
    user_input = conversation.ConversationInput(
        text=f"Fahre {entity_id} auf {value}% für {duration}",
        context=MagicMock(),
        conversation_id=f"test_timebox_{entity_id}",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassSetPosition"
    mock_match.entities = {
        "name": MagicMock(value="buro_nord"),
        "position": MagicMock(value=value),
        "duration": MagicMock(value=duration),
    }

    # Mock script existence directly in IntentExecutorCapability
    from multistage_assist.capabilities.intent_executor import IntentExecutorCapability
    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match), \
         patch.object(IntentExecutorCapability, "_check_script_exists", return_value=True):
        result = await agent.async_process(user_input)
        assert result is not None
        script_calls = [c for c in hass.service_calls if c["domain"] == "script" and "timebox_entity_state" in c["service"]]
        assert len(script_calls) > 0
        call_data = script_calls[0]["service_data"]
        assert call_data["target_entity"] == entity_id
        assert call_data["value"] == value
