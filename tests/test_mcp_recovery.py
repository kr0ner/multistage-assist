import pytest
from unittest.mock import AsyncMock, Mock, patch, call
import json
import sys
import os

sys.path.insert(0, os.getcwd())

from multistage_assist.stage2_llm import Stage2LLMProcessor
from multistage_assist.capabilities.mcp import McpToolCapability
from multistage_assist.const import CONF_STAGE1_IP, CONF_STAGE1_PORT, CONF_STAGE1_MODEL

# Minimal mocks
class MockInput:
    def __init__(self, text):
        self.text = text
        self.context = {}
        self.conversation_id = "test_conv"
        self.language = "en"

@pytest.mark.asyncio
class TestMcpToolCapability:
    
    async def test_resolve_entity_via_llm_flow(self):
        """Test the ReAct loop inside the capability."""
        hass = Mock()
        mcp = McpToolCapability(hass, {})
        
        # Mock internal tools in the tools registry
        # The capability uses self.tools[name].execute
        list_entities_tool = AsyncMock()
        list_entities_tool.execute = AsyncMock(return_value=[{"entity_id": "light.kitchen_main"}])
        mcp.tools = {"list_entities": list_entities_tool}
        
        # Other mocks for the loop
        mcp.get_tools = Mock(return_value=[{"name": "list_entities"}])
        mcp.get_entity_details = AsyncMock(return_value={})
        
        # Patch OllamaClient where it is imported (inside resolve_entity_via_llm)
        with patch("multistage_assist.ollama_client.OllamaClient") as client_cls:
            client_inst = AsyncMock()
            client_cls.return_value = client_inst
            
            # Sequence: 1. Tool Call -> 2. Final Answer
            client_inst.chat_completion.side_effect = [
                json.dumps({"tool": "list_entities", "args": {"area_name": "kitchen"}}),
                json.dumps({"final_answer": ["light.kitchen_main"]})
            ]
            
            llm_config = {"ip": "127.0.0.1", "port": 11434, "model": "test"}
            result = await mcp.resolve_entity_via_llm("text", {}, "intent", "domain", llm_config)
            
            assert result == ["light.kitchen_main"]
            
            # Verify tool execution via the tool's execute method
            list_entities_tool.execute.assert_called_with(area_name="kitchen")

@pytest.mark.asyncio
class TestStage2CallsMcp:
    
    async def test_stage2_delegates_to_mcp(self):
        """Test Stage2 calls mcp.resolve_entity_via_llm when resolution fails."""
        hass = Mock()
        config = {CONF_STAGE1_IP: "1.2.3.4", CONF_STAGE1_PORT: 1234, CONF_STAGE1_MODEL: "model"}
        stage = Stage2LLMProcessor(hass, config)
        
        mcp_mock = Mock()
        mcp_mock.resolve_entity_via_llm = AsyncMock(return_value=["light.test"])
        
        resolver_mock = AsyncMock()
        resolver_mock.run.return_value = {"resolved_ids": []}
        
        ki_data = {"intent": "X", "domain": "Y", "slots": {}}
        
        # Mock defaults
        def get_side_effect(n):
            if n == "mcp_tool": return mcp_mock
            if n == "entity_resolver": return resolver_mock
            if n == "clarification": return None
            return AsyncMock()

        stage.get = Mock(side_effect=get_side_effect)
        stage.use = AsyncMock(return_value=ki_data)
        
        inp = MockInput("foo")
        # Must provide commands to avoid escalation
        result = await stage.process(inp, context={"commands": ["foo"]})
        
        assert result.entity_ids == ["light.test"]
        mcp_mock.resolve_entity_via_llm.assert_called_once()
        # Verify config passed
        call_args = mcp_mock.resolve_entity_via_llm.call_args
        assert call_args.args[4]["ip"] == "1.2.3.4"
