"""Tests for semantic cache bypass optimization.

Verifies cache behavior in Stage1CacheProcessor.
"""

from unittest.mock import MagicMock, AsyncMock
import pytest

from multistage_assist.stage1_cache import Stage1CacheProcessor
from multistage_assist.stage_result import StageResult


async def test_cache_miss_escalates(hass, config_entry):
    """Cache miss should escalate to next stage."""
    stage1 = Stage1CacheProcessor(hass, config_entry.data)
    
    # Mock semantic cache capability with miss
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    user_input = MagicMock()
    user_input.text = "Unbekannter Befehl"
    user_input.conversation_id = "test_conv_1"
    
    context = {}
    
    result = await stage1.process(user_input, context)
    
    # Cache lookup SHOULD have been called
    mock_cache.lookup.assert_called_once()
    
    # Should escalate
    assert result.status == "escalate"
    assert result.context.get("cache_miss") is True


async def test_cache_hit_returns_success(hass, config_entry):
    """Cache hit should return success with resolved data."""
    stage1 = Stage1CacheProcessor(hass, config_entry.data)
    
    # Mock semantic cache capability with a hit
    cache_data = {
        "intent": "HassTurnOn",
        "entity_ids": ["light.kuche"],
        "slots": {"area": "Küche", "domain": "light"},
        "score": 0.95,  # score is required
    }
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=cache_data)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    user_input = MagicMock()
    user_input.text = "Licht Küche an"
    user_input.conversation_id = "test_conv_2"
    
    context = {}
    
    result = await stage1.process(user_input, context)
    
    # Should return success with cache data
    assert result.status == "success"
    assert result.intent == "HassTurnOn"
    assert result.entity_ids == ["light.kuche"]
    assert result.context.get("from_cache") is True


async def test_bypass_intents_skip_cache(hass, config_entry):
    """Timer and calendar intents should bypass cache."""
    stage1 = Stage1CacheProcessor(hass, config_entry.data)
    
    # Mock semantic cache - shouldn't be called
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    user_input = MagicMock()
    user_input.text = "Stelle einen Timer"
    user_input.conversation_id = "test_conv_3"
    
    # Context from Stage0 with timer intent
    context = {"nlu_intent": "HassTimerSet"}
    
    result = await stage1.process(user_input, context)
    
    # Should escalate without calling cache
    assert result.status == "escalate"
    assert result.context.get("cache_bypassed") is True
    mock_cache.lookup.assert_not_called()
