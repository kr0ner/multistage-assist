"""Tests for Conversation History context retrieval."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from multistage_assist.capabilities.entity_resolver import EntityResolverCapability

def mock_hass():
    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    return hass

@pytest.mark.asyncio
async def test_entity_resolver_uses_history():
    """Test that entity resolver falls back to history if no entity specified."""
    hass = mock_hass()
    resolver = EntityResolverCapability(hass, {})
    # Mock AreaResolver with AsyncMock since it's awaited
    resolver._area_resolver = AsyncMock()
    resolver._area_resolver.run.return_value = {"match": None}
    
    user_input = MagicMock()
    user_input.text = "Mach es wieder an"
    
    import time
    history = {
        "last_entities": ["light.buero_deckenlampe", "light.buero_stehlampe"],
        "timestamp": time.time()
    }
    
    # We pass history context to run
    result = await resolver.run(
        user_input, 
        entities={"intent": "HassTurnOn"}, 
        history=history
    )
    
    resolved = result.get("resolved_ids", [])
    assert "light.buero_deckenlampe" in resolved
    assert "light.buero_stehlampe" in resolved
    assert len(resolved) == 2

@pytest.mark.asyncio
async def test_entity_resolver_ignores_history_if_explicit():
    """Test that explicit entity overrides history fallback."""
    hass = mock_hass()
    resolver = EntityResolverCapability(hass, {})
    
    # Setup area resolver with AsyncMock
    resolver._area_resolver = AsyncMock()
    area_mock = MagicMock()
    area_mock.id = "wohnzimmer"
    resolver._area_resolver.run.return_value = {"match": "Wohnzimmer"}
    resolver._area_resolver.find_area.return_value = area_mock
    
    # Mock entities in area
    # Note: _entities_in_area is NOT async in the code (it's a helper)
    # Let's check entity_resolver.py to be sure.
    resolver._entities_in_area = MagicMock(return_value=["light.wohnzimmer"])
    
    user_input = MagicMock()
    user_input.text = "Mach das Licht im Wohnzimmer an"
    
    import time
    history = {"last_entities": ["light.buero"], "timestamp": time.time()}
    
    # Because we clearly state "Wohnzimmer", history should be ignored
    result = await resolver.run(
        user_input, 
        entities={"area": "Wohnzimmer", "intent": "HassTurnOn"}, 
        history=history
    )
    
    resolved = result.get("resolved_ids", [])
    assert "light.wohnzimmer" in resolved
    assert "light.buero" not in resolved
