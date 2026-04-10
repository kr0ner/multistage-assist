"""Tests for service domain exposure bypass in EntityResolver.

Service domains like 'notify' represent services, not entities, and can't be
exposed via Settings → Voice Assistants → Expose. The EntityResolver should
bypass exposure filtering for these domains.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Ensure package importable
sys.path.insert(0, os.getcwd())

from multistage_assist.capabilities.entity_resolver import (
    EntityResolverCapability,
)
from multistage_assist.const import SERVICE_DOMAINS


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.states.async_all.return_value = []
    return hass


@pytest.fixture
def resolver(mock_hass):
    """Create EntityResolverCapability with mocked dependencies."""
    resolver = EntityResolverCapability(mock_hass, {})
    resolver._area_resolver = MagicMock()
    resolver._area_resolver.find_area.return_value = None
    resolver._area_resolver.find_floor.return_value = None
    return resolver


class TestServiceDomainsBypass:
    """Tests for SERVICE_DOMAINS constant and exposure bypass."""
    
    def test_notify_in_service_domains(self):
        """Verify notify domain is in SERVICE_DOMAINS."""
        assert "notify" in SERVICE_DOMAINS
    
    @pytest.mark.asyncio
    async def test_notify_bypasses_exposure_filter(self, resolver, mock_hass):
        """Test that notify.* entities bypass exposure filtering."""
        # Setup: Create notify entity that is NOT exposed
        mock_state = MagicMock()
        mock_state.entity_id = "notify.mobile_app_phone"
        mock_state.attributes = {"friendly_name": "Mobile App Phone"}
        mock_hass.states.get.return_value = mock_state
        mock_hass.states.async_all.return_value = [mock_state]
        
        # Mock _collect_all_domain_entities to return the notify entity
        resolver._collect_all_domain_entities = MagicMock(
            return_value=["notify.mobile_app_phone"]
        )
        
        # Mock async_should_expose to return False (entity not exposed)
        with patch(
            "multistage_assist.capabilities.entity_resolver.async_should_expose",
            return_value=False
        ):
            user_input = MagicMock()
            user_input.text = "Stelle alle Timer"
            
            # Request all entities from notify domain
            slots = {"domain": "notify"}
            
            result = await resolver.run(user_input, entities=slots)
            
            # Even though async_should_expose returns False,
            # notify entities should NOT be filtered out
            assert "notify.mobile_app_phone" in result["resolved_ids"], \
                "notify.* entities should bypass exposure filtering"
    
    @pytest.mark.asyncio
    async def test_non_service_domain_respects_exposure(self, resolver, mock_hass):
        """Test that non-service domains still respect exposure filtering."""
        # Setup: Create light entity that is NOT exposed
        mock_state = MagicMock()
        mock_state.entity_id = "light.living_room"
        mock_state.attributes = {"friendly_name": "Living Room Light"}
        mock_hass.states.get.return_value = mock_state
        mock_hass.states.async_all.return_value = [mock_state]
        
        resolver._collect_all_domain_entities = MagicMock(
            return_value=["light.living_room"]
        )
        
        # Mock async_should_expose to return False (entity not exposed)
        with patch(
            "multistage_assist.capabilities.entity_resolver.async_should_expose",
            return_value=False
        ):
            user_input = MagicMock()
            user_input.text = "Schalte alle Lichter an"
            
            slots = {"domain": "light"}
            
            result = await resolver.run(user_input, entities=slots)
            
            # Light entity should be filtered out because it's not exposed
            assert "light.living_room" not in result["resolved_ids"], \
                "Non-service domain entities should respect exposure filtering"
            assert "light.living_room" in result["filtered_not_exposed"], \
                "Filtered entity should be in filtered_not_exposed list"


class TestMultipleNotifyServices:
    """Tests for discovering multiple notify services."""
    
    @pytest.mark.asyncio
    async def test_discovers_all_notify_services(self, resolver, mock_hass):
        """Test that all notify.mobile_app_* services are discovered."""
        # Setup: Multiple notify services
        states = []
        for name in ["phone", "tablet", "watch"]:
            state = MagicMock()
            state.entity_id = f"notify.mobile_app_{name}"
            state.attributes = {"friendly_name": f"Mobile App {name.title()}"}
            states.append(state)
        
        mock_hass.states.async_all.return_value = states
        mock_hass.states.get.side_effect = lambda eid: next(
            (s for s in states if s.entity_id == eid), None
        )
        
        resolver._collect_all_domain_entities = MagicMock(
            return_value=[s.entity_id for s in states]
        )
        
        # Mock async_should_expose to return False for all
        with patch(
            "multistage_assist.capabilities.entity_resolver.async_should_expose",
            return_value=False
        ):
            user_input = MagicMock()
            user_input.text = "Timer alle Geräte"
            
            slots = {"domain": "notify"}
            
            result = await resolver.run(user_input, entities=slots)
            
            # All notify services should be discovered
            assert len(result["resolved_ids"]) == 3
            assert "notify.mobile_app_phone" in result["resolved_ids"]
            assert "notify.mobile_app_tablet" in result["resolved_ids"]
            assert "notify.mobile_app_watch" in result["resolved_ids"]
