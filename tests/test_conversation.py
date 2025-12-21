"""Test the conversation agent."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from homeassistant.components import conversation
from homeassistant.helpers import intent

from multistage_assist.conversation import MultiStageAssistAgent
from multistage_assist.stage_result import StageResult
from multistage_assist.execution_pipeline import ExecutionResult


@pytest.fixture
def mock_stages():
    """Mock the stages with process() interface."""
    stage0 = MagicMock()
    stage0.process = AsyncMock()
    stage0.has_pending = MagicMock(return_value=False)

    stage1 = MagicMock()
    stage1.process = AsyncMock()
    stage1.has_pending = MagicMock(return_value=False)
    stage1.has = MagicMock(return_value=False)

    stage2 = MagicMock()
    stage2.process = AsyncMock()
    stage2.has_pending = MagicMock(return_value=False)

    stage3 = MagicMock()
    stage3.process = AsyncMock()
    stage3.has_pending = MagicMock(return_value=False)

    return [stage0, stage1, stage2, stage3]


async def test_pipeline_success_stage0(hass, config_entry, mock_stages):
    """Test pipeline handling by stage 0."""
    agent = MultiStageAssistAgent(hass, config_entry.data)
    agent.stages = mock_stages

    user_input = conversation.ConversationInput(
        text="Turn on the light",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="en",
    )

    # Stage 0 returns success with intent
    resp = intent.IntentResponse(language="en")
    mock_stages[0].process.return_value = StageResult.success(
        intent="HassTurnOn",
        entity_ids=["light.test"],
        params={},
        context={},
        raw_text="Turn on the light",
    )

    # Mock execution pipeline
    with patch.object(agent._execution_pipeline, 'execute') as mock_exec:
        mock_exec.return_value = ExecutionResult(
            success=True,
            response=conversation.ConversationResult(response=resp),
            pending_data=None,
        )
        
        result = await agent.async_process(user_input)

        assert result is not None
        mock_stages[0].process.assert_called_once()
        mock_exec.assert_called_once()


async def test_pipeline_escalation(hass, config_entry, mock_stages):
    """Test pipeline escalation."""
    agent = MultiStageAssistAgent(hass, config_entry.data)
    agent.stages = mock_stages

    user_input = conversation.ConversationInput(
        text="Complex query",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="en",
    )

    # Stage 0 escalates
    mock_stages[0].process.return_value = StageResult.escalate(
        context={"test": "data"},
        raw_text="Complex query",
    )
    # Stage 1 returns success
    resp = intent.IntentResponse(language="en")
    mock_stages[1].process.return_value = StageResult.success(
        intent="HassGetState",
        entity_ids=["sensor.test"],
        params={},
        context={},
        raw_text="Complex query",
    )

    with patch.object(agent._execution_pipeline, 'execute') as mock_exec:
        mock_exec.return_value = ExecutionResult(
            success=True,
            response=conversation.ConversationResult(response=resp),
            pending_data=None,
        )
        
        result = await agent.async_process(user_input)

        assert result is not None
        mock_stages[0].process.assert_called_once()
        mock_stages[1].process.assert_called_once()


async def test_fallback(hass, config_entry, mock_stages):
    """Test fallback to default agent when all stages fail/escalate."""
    agent = MultiStageAssistAgent(hass, config_entry.data)
    agent.stages = mock_stages

    user_input = conversation.ConversationInput(
        text="Unknown command",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="en",
    )

    # All stages escalate
    for stage in mock_stages:
        stage.process.return_value = StageResult.escalate(
            context={},
            raw_text="Unknown command",
        )

    with patch(
        "multistage_assist.conversation.conversation.async_converse"
    ) as mock_converse:
        mock_converse.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en")
        )

        result = await agent.async_process(user_input)

        assert result is not None
        mock_converse.assert_called_once()
