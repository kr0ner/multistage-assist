"""Integration tests for MCP parameter passing."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation
from multistage_assist.stage2_llm import Stage2LLMProcessor
from multistage_assist.capabilities.mcp import McpToolCapability
from multistage_assist.capabilities.entity_resolver import EntityResolverCapability

pytestmark = pytest.mark.integration

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    return hass

@pytest.fixture
def stage2_llm(mock_hass):
    config = {
        "stage1_ip": "127.0.0.1",
        "stage1_port": 11434,
        "stage1_model": "qwen3:4b-q8_0"
    }
    processor = Stage2LLMProcessor(mock_hass, config)
    
    # Mock capabilities
    mcp_mock = AsyncMock(spec=McpToolCapability)
    mcp_mock.name = "mcp_tool"
    # Make resolve return correct format
    mcp_mock.resolve_entity_via_llm.return_value = ["light.recovered_via_mcp"]
    
    er_mock = AsyncMock(spec=EntityResolverCapability)
    er_mock.name = "entity_resolver"
    er_mock.run.return_value = {"resolved_ids": []} # Simulate failure
    
    ki_mock = AsyncMock()
    ki_mock.name = "keyword_intent"
    ki_mock.run.return_value = {
        "intent": "HassTurnOn",
        "domain": "light",
        "slots": {"name": "weird light"}
    }
    
    processor.capabilities = [MagicMock, MagicMock, MagicMock] # Just placeholders
    # Inject mocks manually into internal registry if needed, 
    # but Stage2 uses self.get("name"). 
    # We need to mock .get() or setup capabilities properly.
    
    # Easiest way: patch self.get
    processor.get = MagicMock(side_effect=lambda name: {
        "mcp_tool": mcp_mock,
        "entity_resolver": er_mock,
        "keyword_intent": ki_mock,
        "clarification": AsyncMock(run=AsyncMock(return_value=["Turn on weird light"]))
    }.get(name))
    
    # Patch self.use to return the result of ki_mock.run()
    # Since processor.use is async, the side_effect should return the value directly if it's not a coroutine 
    # or be an async function.
    # But AsyncMock automatically wraps return values in coroutines.
    # The issue was likely that ki_mock.run() returns a coroutine (because it's an AsyncMock child)
    # and use also wraps it, or the side_effect logic was flawed.
    
    async def mock_use(name, *args, **kwargs):
        if name == "keyword_intent":
            return {
                "intent": "HassTurnOn",
                "domain": "light",
                "slots": {"name": "weird light"}
            }
        return {}

    processor.use = AsyncMock(side_effect=mock_use)
        
    return processor, mcp_mock

@pytest.mark.asyncio
async def test_mcp_params_passed_correctly(stage2_llm):
    """Test that correct params are passed to MCP capability."""
    processor, mcp_mock = stage2_llm
    
    user_input = conversation.ConversationInput(
        text="Turn on weird light",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="en"
    )
    
    # Run processor
    result = await processor.process(user_input)
    
    # Verify MCP was called
    assert mcp_mock.resolve_entity_via_llm.called
    
    # Verify arguments
    call_args = mcp_mock.resolve_entity_via_llm.call_args
    # signature: resolve_entity_via_llm(text, slots, intent, domain, llm_config)
    
    args, _ = call_args
    assert args[0] == "Turn on weird light"
    assert args[1] == {"name": "weird light"} # slots
    assert args[2] == "HassTurnOn" # intent
    assert args[3] == "light" # domain
    
    llm_config = args[4]
    assert llm_config["ip"] == "127.0.0.1"
    assert llm_config["port"] == 11434
    assert llm_config["model"] == "qwen3:4b-q8_0"
    
    # Verify result uses recovered ID
    assert result.entity_ids == ["light.recovered_via_mcp"]
