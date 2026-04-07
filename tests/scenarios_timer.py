"""Test scenarios for Timer domain."""

import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation
from homeassistant.helpers import intent

from multistage_assist.conversation import MultiStageAssistAgent
from multistage_assist.stage0 import Stage0Processor
from multistage_assist.stage1_cache import Stage1CacheProcessor
from multistage_assist.stage2_llm import Stage2LLMProcessor
from multistage_assist.stage3_cloud import Stage3CloudProcessor
from multistage_assist import conversation_utils
from multistage_assist.stage_result import StageResult

_LOGGER = logging.getLogger(__name__)

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

    # Mock IntentExecutor.run to avoid real Hass calls
    from multistage_assist.capabilities.intent_executor import IntentExecutorCapability
    async def mock_executor_run(user_input, intent_name, entity_ids, params):
        from homeassistant.helpers import intent
        resp = intent.IntentResponse(language="de")
        if intent_name == "HassTimerSet":
            duration = params.get("duration", "unbekannte Zeit")
            resp.async_set_speech(f"Timer für {duration} wurde gestartet.")
        else:
            resp.async_set_speech("Okay.")
        return {"status": "success", "result": conversation.ConversationResult(response=resp)}
        
    patcher_exec = patch.object(IntentExecutorCapability, "run", side_effect=mock_executor_run)
    patcher_exec.start()

    # Mock Stage 2 (Local LLM)
    from multistage_assist.stage_result import StageResult
    async def mock_stage2_process(user_input, context):
        text = user_input.text.lower()
        _LOGGER.debug(f"[Mock] Stage 2 processing: '{text}' (context keys: {list((context or {}).keys())})")
        
        if "timer" in text or "minuten" in text or "sekunden" in text:
            # Simple duration extraction for testing
            duration = None
            if "5 minut" in text:
                duration = "5 Minuten"
            elif "30 sekund" in text:
                duration = "30 Sekunden"
            
            if not duration and "timer" in text and not any(k in text for k in ["minut", "sekund"]):
                # Turn 1: No duration - ask user
                from custom_components.multistage_assist.conversation_utils import make_response
                return StageResult.pending(
                    pending_type="slot_filling",
                    message="Wie lange soll der Timer laufen?",
                    pending_data={"intent": "HassTimerSet", "candidates": {}}
                )

            if duration:
                # Turn 2: Success
                _LOGGER.debug(f"[Mock] Stage 2 resolved timer with duration: {duration}")
                return StageResult.success(
                    intent="HassTimerSet",
                    entity_ids=["timer.default"],
                    params={"duration": duration},
                    context={"from_llm": True}
                )
            
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
    patcher_exec.stop()
    patcher.stop()

async def test_scenario_timer_multi_turn(agent, hass):
    """Scenario 7: Timer - Multi-turn (no duration initially)."""
    # Turn 1
    user_input1 = conversation.ConversationInput(
        text="Stelle einen Timer",
        context=MagicMock(),
        conversation_id="test_id_7",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result1 = await agent.async_process(user_input1)
        assert result1 is not None
        speech1 = result1.response.speech["plain"]["speech"]
        assert "wie lange" in speech1.lower() or "dauer" in speech1.lower()
        assert "test_id_7" in agent._execution_pending

    # Turn 2
    user_input2 = conversation.ConversationInput(
        text="5 Minuten",
        context=MagicMock(),
        conversation_id="test_id_7",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result2 = await agent.async_process(user_input2)
        assert result2 is not None
        speech2 = result2.response.speech["plain"]["speech"]
        assert "timer" in speech2.lower()
        assert "5 minuten" in speech2.lower()

async def test_scenario_timer_with_memory(agent, hass):
    """Scenario 8: Timer with Device Name (using memory)."""
    memory_cap = agent.stages[1].get("memory")
    await memory_cap._ensure_loaded()
    memory_cap._data["entities"]["daniel's handy"] = "notify.mobile_app_sm_a566b"

    # Turn 1
    user_input1 = conversation.ConversationInput(
        text="Stelle einen Timer auf Daniel's Handy",
        context=MagicMock(),
        conversation_id="test_id_8",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result1 = await agent.async_process(user_input1)
        assert result1 is not None
        speech1 = result1.response.speech["plain"]["speech"]
        assert "wie lange" in speech1.lower() or "dauer" in speech1.lower()

    # Turn 2
    user_input2 = conversation.ConversationInput(
        text="30 Sekunden",
        context=MagicMock(),
        conversation_id="test_id_8",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result2 = await agent.async_process(user_input2)
        assert result2 is not None
        speech2 = result2.response.speech["plain"]["speech"]
        assert "timer" in speech2.lower()
        assert "30 sekunden" in speech2.lower()
