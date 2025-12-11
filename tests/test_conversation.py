"""Test the conversation agent."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from homeassistant.components import conversation

from multistage_assist.conversation import MultiStageAssistAgent


@pytest.fixture
def mock_stages():
    """Mock the stages."""
    stage0 = MagicMock()
    stage0.run = AsyncMock()
    stage0.has_pending = MagicMock(return_value=False)

    stage1 = MagicMock()
    stage1.run = AsyncMock()
    stage1.has_pending = MagicMock(return_value=False)

    stage2 = MagicMock()
    stage2.run = AsyncMock()
    stage2.has_pending = MagicMock(return_value=False)

    return [stage0, stage1, stage2]


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

    # Stage 0 handles it
    mock_stages[0].run.return_value = {
        "status": "handled",
        "result": conversation.ConversationResult(
            response=intent.IntentResponse(language="en")
        ),
    }

    result = await agent.async_process(user_input)

    assert result is not None
    mock_stages[0].run.assert_called_once()
    mock_stages[1].run.assert_not_called()


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
    mock_stages[0].run.return_value = {
        "status": "escalate",
        "result": "intermediate_result",
    }
    # Stage 1 handles
    mock_stages[1].run.return_value = {
        "status": "handled",
        "result": conversation.ConversationResult(
            response=intent.IntentResponse(language="en")
        ),
    }

    result = await agent.async_process(user_input)

    assert result is not None
    mock_stages[0].run.assert_called_once()
    mock_stages[1].run.assert_called_once_with(user_input, "intermediate_result")
    mock_stages[2].run.assert_not_called()


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

    # All stages escalate or return None
    mock_stages[0].run.return_value = {"status": "escalate", "result": None}
    mock_stages[1].run.return_value = {"status": "escalate", "result": None}
    mock_stages[2].run.return_value = {"status": "escalate", "result": None}

    with patch(
        "multistage_assist.conversation.conversation.async_converse"
    ) as mock_converse:
        mock_converse.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en")
        )

        result = await agent.async_process(user_input)

        assert result is not None
        mock_converse.assert_called_once()


from homeassistant.helpers import intent
