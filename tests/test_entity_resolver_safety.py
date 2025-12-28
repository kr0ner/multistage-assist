import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Ensure package importable
sys.path.insert(0, os.getcwd())

from multistage_assist.capabilities.entity_resolver import EntityResolverCapability

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.states.async_all.return_value = []
    return hass

@pytest.mark.asyncio
async def test_global_fallback_safety_no_keyword(mock_hass):
    """Test that global fallback is SKIPPED if no 'all' keyword is present."""
    resolver = EntityResolverCapability(mock_hass, {})
    
    # Mock helpers
    resolver._area_resolver = MagicMock()
    resolver._area_resolver.find_area.return_value = None
    resolver._area_resolver.find_floor.return_value = None
    resolver._collect_all_domain_entities = MagicMock(return_value=["light.one", "light.two"])
    
    # Mock GENERIC_NAMES to include "spots" (simulating it being generic)
    with patch("multistage_assist.capabilities.entity_resolver.GENERIC_NAMES", {"spots", "light"}):
        # Input: "Schalte die Spots an" (NO 'alle')
        user_input = MagicMock()
        user_input.text = "Schalte die Spots an"
        
        # Slots indicate domain light, name spots
        slots = {"domain": "light", "name": "spots"}
        
        result = await resolver.run(user_input, entities=slots)
        
        # Expect: "spots" -> treated as generic (name=None).
        # Fallback check -> No "alle" -> Fallback skipped.
        # Result -> Empty
        assert result["resolved_ids"] == [], "Should return empty list for generic name without 'all' keyword"

@pytest.mark.asyncio
async def test_global_fallback_safety_with_keyword(mock_hass):
    """Test that global fallback is ALLOWED if 'all' keyword is present."""
    resolver = EntityResolverCapability(mock_hass, {})
    
    # Mock helpers
    resolver._area_resolver = MagicMock()
    resolver._area_resolver.find_area.return_value = None
    resolver._area_resolver.find_floor.return_value = None
    resolver._collect_all_domain_entities = MagicMock(return_value=["light.one", "light.two"])
    
    # Mock GENERIC_NAMES
    with patch("multistage_assist.capabilities.entity_resolver.GENERIC_NAMES", {"spots", "light"}):
        # Input: "Schalte ALLE Spots an"
        user_input = MagicMock()
        user_input.text = "Schalte alle Spots an"
        
        slots = {"domain": "light", "name": "spots"}
        
        result = await resolver.run(user_input, entities=slots)
        
        # Expect: "spots" -> generic.
        # Fallback check -> Has "alle" -> Fallback executed.
        assert "light.one" in result["resolved_ids"]
        assert "light.two" in result["resolved_ids"]
