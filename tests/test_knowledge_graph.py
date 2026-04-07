"""Tests for the KnowledgeGraphCapability module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from multistage_assist.capabilities.knowledge_graph import (
    KnowledgeGraphCapability,
    RelationType,
    ActivationMode,
    Dependency,
    DependencyResolution,
)

@pytest.fixture
def hass():
    """Create a mock Home Assistant instance with entities."""
    hass = MagicMock()
    
    # Create mock states with attributes
    states = {}
    
    # Kitchen radio with power dependency
    radio_state = MagicMock()
    radio_state.entity_id = "media_player.kitchen_radio"
    radio_state.state = "off"
    radio_state.attributes = {
        "friendly_name": "Kitchen Radio",
        "powered_by": "switch.kitchen_main_power",
        "activation_mode": "auto",
    }
    states["media_player.kitchen_radio"] = radio_state
    
    # Kitchen power switch (off)
    power_state = MagicMock()
    power_state.entity_id = "switch.kitchen_main_power"
    power_state.state = "off"
    power_state.attributes = {"friendly_name": "Kitchen Power"}
    states["switch.kitchen_main_power"] = power_state
    
    # Set up hass.states
    def get_state(entity_id):
        return states.get(entity_id)
    
    hass.states.get = get_state
    hass.states.async_all = MagicMock(return_value=list(states.values()))
    
    return hass

@pytest.fixture
def config():
    return {}

@pytest.mark.asyncio
class TestKnowledgeGraphCapability:
    """Tests for KnowledgeGraphCapability class."""
    
    async def test_get_dependencies(self, hass, config):
        """Test getting dependencies for an entity."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Should detect the power dependency from attributes
        deps = capability.get_dependencies("media_player.kitchen_radio")
        assert len(deps) == 1
        assert deps[0].relation_type == RelationType.POWERED_BY
        assert deps[0].target_entity == "switch.kitchen_main_power"
        assert deps[0].activation_mode == ActivationMode.AUTO

    async def test_resolve_for_action_auto(self, hass, config):
        """Test resolving dependency with auto activation."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Test turn_on action
        resolution = await capability.resolve_for_action("media_player.kitchen_radio", "turn_on")
        
        assert resolution.can_proceed is True
        assert len(resolution.prerequisites) == 1
        assert resolution.prerequisites[0]["entity_id"] == "switch.kitchen_main_power"
        assert resolution.prerequisites[0]["action"] == "turn_on"

    async def test_hardcoded_overrides(self, hass, config):
        """Test that hardcoded overrides (BUG 22) are loaded."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Add mock state for the hardcoded entity
        tv_state = MagicMock()
        tv_state.entity_id = "media_player.ue55j6250"
        tv_state.state = "off"
        hass.states.get.side_effect = lambda eid: tv_state if eid == "media_player.ue55j6250" else None
        
        deps = capability.get_dependencies("light.h6099")
        assert len(deps) >= 1
        assert any(d.target_entity == "media_player.ue55j6250" for d in deps)

    async def test_alias_learning(self, hass, config):
        """Test alias learning and retrieval (Legacy Memory functionality)."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Mock storage
        capability._store = MagicMock()
        capability._store.async_load = AsyncMock(return_value={"areas": {}, "entities": {}, "floors": {}, "personal": {}, "relationships": {}})
        capability._store.async_save = AsyncMock()
        
        await capability.learn_area_alias("bad", "Badezimmer")
        
        alias = await capability.get_area_alias("bad")
        assert alias == "Badezimmer"
        
        # Case insensitive retrieval
        assert await capability.get_area_alias("BAD") == "Badezimmer"

    async def test_personal_data(self, hass, config):
        """Test personal data storage and retrieval."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Mock storage
        capability._store = MagicMock()
        capability._store.async_load = AsyncMock(return_value={"areas": {}, "entities": {}, "floors": {}, "personal": {"name": "Daniel"}, "relationships": {}})
        
        name = await capability.get_personal_data("name")
        assert name == "Daniel"
        
        all_data = await capability.get_all_personal_data()
        assert all_data["name"] == "Daniel"
