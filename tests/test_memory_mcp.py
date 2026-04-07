"""Tests for personal memory capability and MCP exposed tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from multistage_assist.capabilities.knowledge_graph import KnowledgeGraphCapability
from multistage_assist.capabilities.mcp import McpToolCapability

@pytest.mark.asyncio
async def test_memory_personal_data():
    hass = MagicMock()
    # Mock Store
    memory = KnowledgeGraphCapability(hass, {})
    memory._store = MagicMock()
    memory._store.async_load = AsyncMock(return_value={"personal": {}})
    memory._store.async_save = AsyncMock()
    
    await memory.learn_personal_data("katze", "mimi")
    val = await memory.get_personal_data("katze")
    assert val == "mimi"
    
    data = await memory.get_all_personal_data()
    assert data["katze"] == "mimi"

@pytest.mark.asyncio
async def test_mcp_personal_memory_tools():
    hass = MagicMock()
    mcp = McpToolCapability(hass, {})
    
    # Setup mocked memory
    memory = KnowledgeGraphCapability(hass, {})
    memory.learn_personal_data = AsyncMock()
    memory.get_personal_data = AsyncMock(return_value="daniel")
    
    mcp.set_memory(memory)
    
    # Store data tool
    res = await mcp.execute_tool("store_personal_data", {"key": "name", "value": "daniel"})
    assert res == {"status": "success", "message": "Saved 'name'"}
    memory.learn_personal_data.assert_called_once_with("name", "daniel")
    
    # Retrieve data tool
    res2 = await mcp.execute_tool("get_personal_data", {"key": "name"})
    assert res2 == {"key": "name", "value": "daniel"}
