"""Tests for NLU passthrough optimization in Stage1.

Verifies that semantic cache is NOT consulted when Stage0 already resolved intent.
"""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from multistage_assist.stage1 import Stage1Processor
from multistage_assist.stage_result import Stage0Result


async def test_cache_skipped_when_stage0_has_intent(hass, config_entry):
    """Cache lookup should be skipped if Stage0 already resolved intent."""
    stage1 = Stage1Processor(hass, config_entry.data)
    
    # Mock semantic cache capability
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    # Mock other capabilities to avoid errors
    stage1.capabilities_map["command_processor"] = MagicMock()
    stage1.capabilities_map["command_processor"].process = AsyncMock(return_value={
        "status": "success",
        "response": MagicMock()
    })
    stage1.capabilities_map["intent_resolution"] = MagicMock()
    stage1.capabilities_map["intent_resolution"].run = AsyncMock(return_value=None)
    
    # Create Stage0Result WITH intent (NLU success)
    prev_result = Stage0Result(
        type="nlu",
        resolved_ids=["light.kuche", "light.kuche_spots"],
        intent="HassTurnOn",
        slots={"area": "Küche", "domain": "light"},
        params={}
    )
    
    user_input = MagicMock()
    user_input.text = "Schalte das Licht in der Küche an"
    user_input.conversation_id = "test_conv_1"
    
    # Run Stage1 with Stage0Result
    try:
        await stage1.run(user_input, prev_result=prev_result)
    except Exception:
        pass  # We only care about cache lookup
    
    # Cache lookup should NOT have been called
    mock_cache.lookup.assert_not_called()


async def test_cache_consulted_when_no_stage0_intent(hass, config_entry):
    """Cache lookup should happen when Stage0 didn't resolve intent."""
    stage1 = Stage1Processor(hass, config_entry.data)
    
    # Mock semantic cache capability
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    # Mock other capabilities
    stage1.capabilities_map["clarification"] = MagicMock()
    stage1.capabilities_map["clarification"].run = AsyncMock(return_value=None)
    
    user_input = MagicMock()
    user_input.text = "Mach das Ding an"  # Vague command, no NLU match
    user_input.conversation_id = "test_conv_2"
    
    # Run Stage1 WITHOUT prev_result (no Stage0 intent)
    try:
        await stage1.run(user_input, prev_result=None)
    except Exception:
        pass  # We only care about cache lookup
    
    # Cache lookup SHOULD have been called
    mock_cache.lookup.assert_called_once_with("Mach das Ding an")


async def test_cache_consulted_when_stage0_no_intent(hass, config_entry):
    """Cache lookup should happen when Stage0Result has no intent."""
    stage1 = Stage1Processor(hass, config_entry.data)
    
    # Mock semantic cache
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock(return_value=None)
    stage1.capabilities_map["semantic_cache"] = mock_cache
    
    # Mock other capabilities
    stage1.capabilities_map["clarification"] = MagicMock()
    stage1.capabilities_map["clarification"].run = AsyncMock(return_value=None)
    
    # Stage0Result WITHOUT intent (NLU failed to detect)
    prev_result = Stage0Result(
        type="nlu",
        resolved_ids=[],
        intent=None,  # No intent detected
        slots={},
        params={}
    )
    
    user_input = MagicMock()
    user_input.text = "Mach das Ding an"
    user_input.conversation_id = "test_conv_3"
    
    # Run Stage1 with Stage0Result but no intent
    try:
        await stage1.run(user_input, prev_result=prev_result)
    except Exception:
        pass
    
    # Cache lookup SHOULD have been called (no NLU intent)
    mock_cache.lookup.assert_called_once()

