import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation
from multistage_assist.stage1_cache import Stage1CacheProcessor
from multistage_assist.capabilities.atomic_command import AtomicCommandCapability
from multistage_assist.capabilities.implicit_intent import ImplicitIntentCapability
from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
from multistage_assist.stage3_cloud import Stage3CloudProcessor
from multistage_assist.const import CONF_STAGE3_MODEL

@pytest.mark.asyncio
async def test_atomic_command_splitting():
    """Test splitting of compound commands."""
    hass = MagicMock()
    cap = AtomicCommandCapability(hass, {})
    
    # Mock LLM to simulate splitting (since it uses LLM internally)
    # We mock _safe_prompt to return the split list
    cap._safe_prompt = AsyncMock(return_value=["Licht Küche an", "Licht Flur aus"])
    
    user_input = conversation.ConversationInput(
        text="Licht Küche an und Licht Flur aus",
        context={}, conversation_id="1", device_id="1", language="de"
    )
    
    # Force splitting by using a "compound" looking sentence
    # The heuristic check in run() needs to pass or we force it via mock
    # Actually run() calls _safe_prompt if separators are found. "und" is a separator.
    
    results = await cap.run(user_input)
    assert len(results) == 2
    assert results == ["Licht Küche an", "Licht Flur aus"]

@pytest.mark.asyncio
async def test_implicit_intent_phrasing():
    """Test rephrasing of implicit commands."""
    hass = MagicMock()
    cap = ImplicitIntentCapability(hass, {})
    
    cap._safe_prompt = AsyncMock(return_value=["Mache Licht heller"])
    
    user_input = conversation.ConversationInput(
        text="Es ist zu dunkel",
        context={}, conversation_id="1", device_id="1", language="de"
    )
    
    results = await cap.run(user_input)
    assert results == ["Mache das Licht heller"]

@pytest.mark.asyncio
async def test_gemini_config_model():
    """Verify Stage3 uses the correct config key for model."""
    hass = MagicMock()
    
    # Case 1: Default config (no specific model set)
    config = {"google_api_key": "fake_key"} 
    # Note: Logic falls back to default if key missing
    stage3 = Stage3CloudProcessor(hass, config)
    # stage3 uses config.get(CONF_STAGE3_MODEL, "gemini-1.5-flash")
    # if CONF_STAGE3_MODEL is "stage3_model", and it's missing in config -> default
    assert stage3._gemini_client.model == "gemini-2.5-flash"  # The new default we set
    
    # Case 2: Configured model
    config_explicit = {
        "google_api_key": "fake_key",
        CONF_STAGE3_MODEL: "gemini-pro-configured"
    }
    stage3_explicit = Stage3CloudProcessor(hass, config_explicit)
    assert stage3_explicit._gemini_client.model == "gemini-pro-configured"

@pytest.mark.asyncio
async def test_power_consumption_keyword():
    """Verify 'Stromverbrauch' maps to sensor domain."""
    hass = MagicMock()
    cap = KeywordIntentCapability(hass, {})
    
    # Logic in run() calls _detect_domain
    # We need to verify _detect_domain finds "sensor" for "Stromverbrauch"
    
    user_input = conversation.ConversationInput(
        text="Wie ist der Stromverbrauch?",
        context={}, conversation_id="1", device_id="1", language="de"
    )
    
    # We can test _detect_domain directly or run()
    # run() requires LLM mock for the extraction part
    
    domain = cap._detect_domain(user_input.text)
    assert domain == "sensor"

@pytest.mark.asyncio
async def test_hass_get_state_ambiguity_bypass():
    """Test that HassGetState bypasses strict ambiguity checks."""
    hass = MagicMock()
    
    # Setup Stage1Cache with SemanticCache
    config = {"cache_enabled": True}
    stage1 = Stage1CacheProcessor(hass, config)
    
    # Mock SemanticCacheCapability
    mock_cache = MagicMock()
    mock_cache.lookup = AsyncMock()
    
    # Return ambiguous matches but for HassGetState
    ambiguous_result = {
        "intent": "HassGetState",
        "entity_ids": ["light.kitchen", "light.living_room"],
        "ambiguous_matches": [
             {"intent": "HassGetState", "entity_ids": ["light.kitchen"], "score": 0.8},
             {"intent": "HassGetState", "entity_ids": ["light.living_room"], "score": 0.8}
        ]
    }
    # Wait, the logic is inside semantic_cache.lookup, not stage1. 
    # stage1 just receives the result.
    # I need to test SemanticCacheCapability.lookup()
    
    from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability
    
    sem_cap = SemanticCacheCapability(hass, config)
    sem_cap._addon_url = MagicMock(return_value="http://fake")
    
    # Mock aiohttp response
    # The cache addon returns the best match
    mock_response_data = {
        "found": True,
        "score": 0.95,
        "matches": [
             {"intent": "HassGetState", "entity_ids": ["light.kitchen"], "score": 0.95},
             {"intent": "HassGetState", "entity_ids": ["light.living"], "score": 0.94}
        ]
    }
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value.__aenter__.return_value = mock_resp
        
        result = await sem_cap.lookup("Sind die Lichter an?")
        
        # Should return a single merged result, NOT ambiguous_matches
        assert result["intent"] == "HassGetState"
        assert "ambiguous_matches" not in result
        assert set(result["entity_ids"]) == {"light.kitchen", "light.living"}

@pytest.mark.asyncio
async def test_global_query_exposure_filtering():
    """Test ExecutionPipeline filters out non-exposed entities during global queries."""
    hass = MagicMock()
    
    # Mock hass.states.async_entity_ids to return our entities
    hass.states.async_entity_ids = MagicMock(return_value=["light.kitchen", "light.hidden", "light.script_light"])
    
    from multistage_assist.execution_pipeline import ExecutionPipeline
    from multistage_assist.stage_result import StageResult
    
    pipeline = ExecutionPipeline(hass, {})
    
    # Mock CommandProcessor.process so the test doesn't actually try to execute
    pipeline._processor.process = AsyncMock(return_value={"status": "handled", "result": None})
    
    stage_result = StageResult(
        status="success",
        intent="HassGetState",
        entity_ids=[],  # Empty triggers the global query logic
        params={"domain": "light"},
        context={}
    )
    
    user_input = conversation.ConversationInput(
        text="Welche Lichter sind an?",
        context={}, conversation_id="1", device_id="1", language="de"
    )

    with patch("homeassistant.components.homeassistant.exposed_entities.async_should_expose") as mock_expose:
        # light.kitchen = exposed
        # light.hidden = not exposed
        # light.script_light = exposed but assume it's something we might filter (or let it pass if domain is light)
        
        def mock_should_expose(hass_instance, domain, entity_id):
            if entity_id == "light.kitchen":
                return True
            return False
            
        mock_expose.side_effect = mock_should_expose
        
        await pipeline.execute(user_input, stage_result)
        
        # After execution pipeline logic, the entity_ids should be exactly ["light.kitchen"]
        assert stage_result.entity_ids == ["light.kitchen"]
        
        # Verify CommandProcessor was called with the filtered list
        pipeline._processor.process.assert_called_once()
        args, kwargs = pipeline._processor.process.call_args
        assert kwargs["candidates"] == ["light.kitchen"]
