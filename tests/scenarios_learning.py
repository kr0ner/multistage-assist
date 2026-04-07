"""Test scenarios for Area Alias Learning."""

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

    # Mock Stage 2 (Local LLM)
    # For Turn 1, we want it to return success + learning_data
    from multistage_assist.stage_result import StageResult
    async def mock_stage2_process(user_input, context):
        if "gästebad" in user_input.text.lower():
            # Return success with learning_data to trigger the offer
            return StageResult.success(
                intent="HassTurnOn",
                entity_ids=["light.gaste_badezimmer"],
                params={"area": "Gäste Badezimmer"},
                context={
                    "learning_data": {
                        "type": "area",
                        "source": "Gästebad",
                        "target": "Gäste Badezimmer"
                    }
                },
                raw_text=user_input.text
            )
        return StageResult.escalate()
    
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

async def test_scenario_area_alias_learning(agent, hass):
    """Scenario 3: Area Alias Learning (Gästebad -> Gäste Badezimmer)."""
    # Turn 1
    user_input = conversation.ConversationInput(
        text="Schalte das Licht im Gästebad an",
        context=MagicMock(),
        conversation_id="test_id_3",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        assert mock_async_handle.called
        assert "light.gaste_badezimmer" in str(mock_async_handle.call_args)
        
        speech = result.response.speech["plain"]["speech"]
        if "merken" in speech.lower() or "alias" in speech.lower():
            assert "test_id_3" in agent._execution_pending
            assert agent._execution_pending["test_id_3"]["type"] == "learning_confirmation"

    # Turn 2: "Ja"
    user_input2 = conversation.ConversationInput(
        text="Ja",
        context=MagicMock(),
        conversation_id="test_id_3",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        agent._execution_pending["test_id_3"] = {
            "type": "learning_confirmation",
            "learning_type": "area",
            "source": "Gästebad",
            "target": "Gäste Badezimmer",
        }
        result2 = await agent.async_process(user_input2)
        assert result2 is not None
        speech = result2.response.speech["plain"]["speech"]
        assert "gemerkt" in speech.lower() or "ok" in speech.lower()
