import pytest
from unittest.mock import MagicMock, AsyncMock
from multistage_assist.stage1_cache import Stage1CacheProcessor
from multistage_assist.stage_result import StageResult
from homeassistant.components import conversation

@pytest.mark.asyncio
async def test_stage1_fallback_entity_resolution():
    """Test that Stage1CacheProcessor triggers entity resolution if cache hit has no entities."""
    # Mock HASS and Config
    hass = MagicMock()
    config = {}
    
    # Input User Text (Define early to avoid UnboundLocalError)
    user_input = conversation.ConversationInput(
        text="Schalte alle Lampen im Haus aus",
        context=None,
        conversation_id="123",
        language="de",
        agent_id="test_agent",
        device_id="test_device_123"
    )
    
    # Mock SemanticCache
    mock_cache = AsyncMock()
    # Cache returns HIT with correct intent/slots but EMPTY entity_ids (Global Anchor simulation)
    mock_cache.lookup.return_value = {
        "intent": "HassTurnOff",
        "entity_ids": [],
        "slots": {"domain": "light"},
        "score": 0.85,
        "source": "anchor"  # or learned
    }
    
    # Mock EntityResolver
    mock_resolver = AsyncMock()
    # Resolver should find entities when called with input
    mock_resolver.run.return_value = {
        "resolved_ids": ["light.living_room", "light.kitchen"],
        "filtered_by_deps": []
    }
    
    # Mock ImplicitIntent
    mock_implicit = AsyncMock()
    mock_implicit.run.return_value = [user_input.text] # No rephrasal
    
    # Mock AtomicCommand
    mock_atomic = AsyncMock()
    mock_atomic.run.return_value = [user_input.text] # No splitting
    
    # Mock Memory
    mock_memory = AsyncMock()
    mock_memory.get_area_alias.return_value = None

    # Setup Processor
    processor = Stage1CacheProcessor(hass, config)
    
    # Inject mocked capabilities logic
    # BaseStage.get returns capabilities by name. We need to mock .get()
    def get_capability(name):
        if name == "semantic_cache": return mock_cache
        if name == "entity_resolver": return mock_resolver
        if name == "implicit_intent": return mock_implicit
        if name == "atomic_command": return mock_atomic
        if name == "knowledge_graph": return mock_memory
        return None
    
    processor.get = MagicMock(side_effect=get_capability)
    processor.has = MagicMock(return_value=True)

    # Run Process
    result = await processor.process(user_input)

    # Assertions
    assert result.status == "success"
    assert result.intent == "HassTurnOff"
    
    # CRITICAL: Entity IDs should be populated from resolver
    assert result.entity_ids == ["light.living_room", "light.kitchen"], \
        "Stage1 failed to populate entity_ids from fallback resolution"
        
    # Verify resolver was called with correct args
    mock_resolver.run.assert_called_once()
    call_args = mock_resolver.run.call_args
    assert call_args[0][0] == user_input # First arg matches user_input
    assert call_args[1]['entities'] == {"domain": "light"} # kwargs['entities'] matches cache slots
