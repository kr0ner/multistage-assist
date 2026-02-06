import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from homeassistant.core import HomeAssistant
from multistage_assist.conversation import MultiStageAssistAgent
from multistage_assist.capabilities.intent_executor import IntentExecutorCapability
from multistage_assist.execution_pipeline import ExecutionPipeline
from multistage_assist.stage_result import StageResult
from multistage_assist.capabilities.command_processor import CommandProcessorCapability

@pytest.mark.asyncio
async def test_global_exit_command(hass: HomeAssistant):
    """Test that 'Abbruch' immediately stops execution."""
    agent = MultiStageAssistAgent(hass, {})
    
    # Mock make_response to verify return value
    with patch("multistage_assist.conversation.make_response", new_callable=AsyncMock) as mock_resp:
        mock_resp.return_value = MagicMock()
        
        # Test "Abbruch"
        input_obj = MagicMock(text="Abbruch", conversation_id="test_exit")
        # Ensure pending state exists to verify cleanup
        agent._execution_pending["test_exit"] = {"some": "state"}
        
        result = await agent.async_process(input_obj)
        
        # Should return "Vorgang abgebrochen."
        mock_resp.assert_called_with("Vorgang abgebrochen.", input_obj)
        assert result == mock_resp.return_value
        # Should clean up pending state
        assert "test_exit" not in agent._execution_pending

@pytest.mark.asyncio
async def test_fraction_normalization(hass: HomeAssistant):
    """Test normalization of 'halb' to 50%."""
    # We test IntentExecutor logic directly
    executor = IntentExecutorCapability(hass, {})
    
    params = {"position": "halb", "brightness": "viertel", "percentage": "dreiviertel"}
    normalized = executor._normalize_params(params)
    
    assert normalized["position"] == 50
    assert normalized["brightness"] == 25
    assert normalized["percentage"] == 75

@pytest.mark.asyncio
async def test_global_query_empty_entities(hass: HomeAssistant):
    """Test global state query fetches all entities."""
    # Mock hass.states
    hass.states.async_entity_ids = MagicMock(return_value=["light.kitchen", "light.bedroom"])
    
    pipeline = ExecutionPipeline(hass, {})
    pipeline._processor = AsyncMock() # Mock command processor
    pipeline._processor.process.return_value = {"status": "handled", "result": MagicMock()}
    
    # Result with NO entities but domain=light
    stage_res = StageResult.success(
        intent="HassGetState",
        entity_ids=[],
        params={"domain": "light", "state": "on"},
        raw_text="Welche Lichter sind an?"
    )
    
    await pipeline.execute(MagicMock(), stage_res)
    
    # Should have fetched entities
    hass.states.async_entity_ids.assert_called_with("light")
    # Verify processor called with ALL entities
    call_args = pipeline._processor.process.call_args
    assert call_args.kwargs["candidates"] == ["light.kitchen", "light.bedroom"]

@pytest.mark.asyncio
async def test_learning_confirmation_flow(hass: HomeAssistant):
    """Test user confirming 'Ja' to learn alias."""
    cp = CommandProcessorCapability(hass, {})
    cp.memory = AsyncMock() # Mock memory
    cp.select = AsyncMock()
    cp.disambiguation = AsyncMock()
    
    # Needs to patch make_response for return value
    with patch("multistage_assist.capabilities.command_processor.make_response", new_callable=AsyncMock) as mock_resp:
        mock_resp.return_value = MagicMock(speech={"plain": {"speech": "Alles klar"}})
        
        # Simulate pending data for learning confirmation
        pending_data = {
            "type": "learning_confirmation",
            "learning_type": "area",
            "source": "Bad",
            "target": "Badezimmer"
        }
        
        # User says "Ja"
        user_input = MagicMock(text="Ja")
        
        res = await cp.continue_disambiguation(user_input, pending_data)
        
        # Should have called memory.learn_area_alias
        cp.memory.learn_area_alias.assert_called_with("Bad", "Badezimmer")
        assert res["status"] == "handled"
