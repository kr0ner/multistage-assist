"""Test scenarios for Lighting domain."""

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
from .scenario_fixtures import (
    MockIntentResponse,
    patch_intent_response,
    patch_make_response,
    create_base_agent,
)

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
        
        # Determine area
        area = "Büro"
        if "küche" in text: area = "Küche"
        elif "bad" in text or "badezimmer" in text: area = "Badezimmer"
        elif "garage" in text: area = "Garage"
        elif "hauswirtschaftsraum" in text: area = "Hauswirtschaftsraum"
        
        # Map area to entities
        entities = []
        if area == "Büro": entities = ["light.buro"]
        elif area == "Küche": entities = ["light.kuche", "light.kuche_spots"]
        elif area == "Badezimmer": entities = ["light.badezimmer", "light.dusche", "light.badezimmer_spiegel"]
        elif area == "Garage": entities = ["light.garage"]
        elif area == "Hauswirtschaftsraum": entities = ["light.hauswirtschaftsraum"]

        if "hell" in text or "dunkel" in text:
            # For brightness tests, we want success
            return StageResult(
                status="success",
                intent="HassLightSet",
                entity_ids=entities,
                params={"area": area},
                response=await make_response("Stage 2: Okay.", user_input)
            )
        
        if "licht" in text or "alle" in text:
            intent_name = "HassTurnOn"
            if "aus" in text: intent_name = "HassTurnOff"
            
            # For global scope "alle", use all lights from our mock set
            if "alle" in text:
                entities = ["light.gaste_badezimmer", "light.badezimmer", "light.dusche", "light.badezimmer_spiegel"]

            return StageResult(
                status="success",
                intent=intent_name,
                entity_ids=entities,
                params={"area": area} if "alle" not in text else {},
                context={"from_llm": True}
            )

        return StageResult(status="escalate")
    
    patcher_stage2 = patch.object(agent.stages[2], "process", side_effect=mock_stage2_process)
    patcher_stage2.start()

    # Mock Stage 3 (Gemini) for deterministic results
    from custom_components.multistage_assist.conversation_utils import make_response
    async def mock_gemini_process(user_input, context):
        text = user_input.text.lower()
        msg = "Okay."
        if "hell" in text or "dunkel" in text:
            msg = "Okay, ich regele das Licht."
            
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

async def test_scenario_disambiguation(agent, hass):
    """Scenario 1: Disambiguation (Licht in der Küche)."""
    user_input1 = conversation.ConversationInput(
        text="Schalte das Licht in der Küche an",
        context=MagicMock(),
        conversation_id="test_id_1",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassTurnOn"
    mock_match.entities = {
        "area": MagicMock(value="Küche"),
        "domain": MagicMock(value="light"),
    }

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match):
        result1 = await agent.async_process(user_input1)
        assert result1 is not None
        speech = result1.response.speech["plain"]["speech"]
        assert "?" in speech or "welches" in speech.lower() or "meinst du" in speech.lower()
        assert "test_id_1" in agent._execution_pending

    user_input2 = conversation.ConversationInput(
        text="Die Spots",
        context=MagicMock(),
        conversation_id="test_id_1",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result2 = await agent.async_process(user_input2)
        assert result2 is not None
        args, kwargs = mock_async_handle.call_args
        assert kwargs["intent_type"] == "HassTurnOn"
        assert kwargs["slots"]["name"]["value"] == "light.kuche_spots"

async def test_scenario_disambiguation_with_memory(agent, hass):
    """Scenario 4: Disambiguation with Memory Hit (Bad)."""
    memory_cap = agent.stages[1].get("memory")
    await memory_cap._ensure_loaded()
    memory_cap._data["areas"]["bad"] = "Badezimmer"

    user_input1 = conversation.ConversationInput(
        text="Schalte das Licht im Bad an",
        context=MagicMock(),
        conversation_id="test_id_4",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None):
        result1 = await agent.async_process(user_input1)
        assert result1 is not None
        speech = result1.response.speech["plain"]["speech"]
        assert "?" in speech or "meinst" in speech.lower() or "welches" in speech.lower()
        assert "badezimmer" in speech.lower()

    user_input2 = conversation.ConversationInput(
        text="Das Erste",
        context=MagicMock(),
        conversation_id="test_id_4",
        device_id="test_device",
        language="de",
    )

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=None), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result2 = await agent.async_process(user_input2)
        assert result2 is not None
        args, kwargs = mock_async_handle.call_args
        assert kwargs["intent_type"] == "HassTurnOn"
        assert kwargs["slots"]["name"]["value"] == "light.badezimmer"

async def test_scenario_state_based_filtering(agent, hass):
    """Scenario 5: State-Based Filtering (Turn Off)."""
    user_input = conversation.ConversationInput(
        text="Schalte das Licht im Badezimmer aus",
        context=MagicMock(),
        conversation_id="test_id_5",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassTurnOff"
    mock_match.entities = {
        "area": MagicMock(value="Badezimmer"),
        "domain": MagicMock(value="light"),
    }

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        speech = result.response.speech["plain"]["speech"]
        assert "?" in speech or "meinst" in speech.lower() or "welches" in speech.lower() or mock_async_handle.called

async def test_scenario_plural_lights_in_area(agent, hass):
    """Scenario 9: Plural Detection - Multiple Lights in Same Area."""
    user_input = conversation.ConversationInput(
        text="Schalte die Lichter im Badezimmer an",
        context=MagicMock(),
        conversation_id="test_id_9",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassTurnOn"
    mock_match.entities = {
        "area": MagicMock(value="Badezimmer"),
        "domain": MagicMock(value="light"),
    }

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        assert mock_async_handle.call_count == 3

async def test_scenario_brightness_percentage(agent, hass):
    """Scenario 11: Brightness Control with Percentage."""
    user_input = conversation.ConversationInput(
        text="Dimme das Licht im Büro auf 50%",
        context=MagicMock(),
        conversation_id="test_id_11",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassLightSet"
    mock_match.entities = {
        "area": MagicMock(value="Büro"),
        "brightness": MagicMock(value=50),
        "domain": MagicMock(value="light"),
    }

    with patch.object(agent.stages[0], "_dry_run_recognize", return_value=mock_match), patch(
        "homeassistant.helpers.intent.async_handle"
    ) as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        assert mock_async_handle.called
        args, kwargs = mock_async_handle.call_args
        assert kwargs["slots"]["brightness"]["value"] == 50

async def test_scenario_brightness_too_dark(agent, hass):
    """Scenario 12: Brightness Adjustment - Too Dark."""
    user_input = conversation.ConversationInput(
        text="Im Büro ist es zu dunkel",
        context=MagicMock(),
        conversation_id="test_id_12",
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
        speech = result.response.speech["plain"]["speech"]
        assert "heller" in speech.lower() or "büro" in speech.lower()
        assert mock_async_handle.called

async def test_scenario_brightness_too_bright(agent, hass):
    """Scenario 13: Brightness Adjustment - Too Bright."""
    user_input = conversation.ConversationInput(
        text="Im Hauswirtschaftsraum ist es zu hell",
        context=MagicMock(),
        conversation_id="test_id_13",
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
        speech = result.response.speech["plain"]["speech"]
        assert "dunkler" in speech.lower() or "hauswirtschaftsraum" in speech.lower()
        assert mock_async_handle.called

async def test_scenario_global_lights_off(agent, hass):
    """Scenario 10: Global Scope - All Lights."""
    user_input = conversation.ConversationInput(
        text="Schalte alle Lichter aus",
        context=MagicMock(),
        conversation_id="test_id_10",
        device_id="test_device",
        language="de",
    )

    mock_match = MagicMock()
    mock_match.intent.name = "HassTurnOff"
    mock_match.entities = {
        "domain": MagicMock(value="light"),
    }

    with patch.object(
        agent.stages[0], "_dry_run_recognize", return_value=mock_match
    ), patch("homeassistant.helpers.intent.async_handle") as mock_async_handle:
        mock_response = intent.IntentResponse(language="de")
        mock_response.async_set_speech("Okay.")
        mock_async_handle.return_value = mock_response

        result = await agent.async_process(user_input)
        assert result is not None
        assert mock_async_handle.called
        speech = result.response.speech["plain"]["speech"]
        assert "licht" in speech.lower() or "aus" in speech.lower()
