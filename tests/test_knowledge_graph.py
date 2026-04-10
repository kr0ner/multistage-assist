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
        
        # Mock storage
        capability._store = MagicMock()
        capability._store.async_load = AsyncMock(return_value={"areas": {}, "entities": {}, "floors": {}, "personal": {}, "relationships": {}})
        
        # Should detect the power dependency from attributes
        deps = await capability.get_dependencies("media_player.kitchen_radio")
        assert len(deps) == 1
        assert deps[0].relation_type == RelationType.POWERED_BY
        assert deps[0].target_entity == "switch.kitchen_main_power"
        assert deps[0].activation_mode == ActivationMode.AUTO

    async def test_resolve_for_action_auto(self, hass, config):
        """Test resolving dependency with auto activation."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Mock storage
        capability._store = MagicMock()
        capability._store.async_load = AsyncMock(return_value={"areas": {}, "entities": {}, "floors": {}, "personal": {}, "relationships": {}})
        
        # Test turn_on action
        resolution = await capability.resolve_for_action("media_player.kitchen_radio", "turn_on")
        
        assert resolution.can_proceed is True
        assert len(resolution.prerequisites) == 1
        assert resolution.prerequisites[0]["entity_id"] == "switch.kitchen_main_power"
        assert resolution.prerequisites[0]["action"] == "turn_on"

    async def test_persistent_dependency(self, hass, config):
        """Test that dependencies from persistent storage are returned."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Simulate a stored relationship
        capability._data = {
            "areas": {}, "entities": {}, "floors": {}, "personal": {},
            "relationships": {
                "light.h6099 -> media_player.ue55j6250": {
                    "source": "light.h6099",
                    "target": "media_player.ue55j6250",
                    "relation": "powered_by",
                    "mode": "auto",
                }
            }
        }
        capability._loaded = True
        
        # No state attributes — dependency comes purely from storage
        hass.states.get.return_value = None
        
        deps = await capability.get_dependencies("light.h6099")
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

    async def test_filter_candidates_by_usability(self, hass, config):
        """Test filtering candidates by their usability (power dependency)."""
        capability = KnowledgeGraphCapability(hass, config)
        
        # Mock storage to avoid actual load
        capability._store = MagicMock()
        capability._store.async_load = AsyncMock(return_value={
            "areas": {}, "entities": {}, "floors": {}, "personal": {}, "relationships": {}
        })
        
        # Scenario 1: Entities without dependencies are usable
        usable, filtered = await capability.filter_candidates_by_usability(["light.living_room"])
        assert usable == ["light.living_room"]
        assert filtered == []
        
        # Scenario 2: Entity with 'auto' dependency is usable
        # (media_player.kitchen_radio is already set up in hass fixture with auto dependency)
        usable, filtered = await capability.filter_candidates_by_usability(["media_player.kitchen_radio"])
        assert "media_player.kitchen_radio" in usable
        
        # Scenario 3: Entity with 'manual' dependency that is OFF is NOT usable
        # We'll mock a new entity for this
        radio_manual_state = MagicMock()
        radio_manual_state.state = "off"
        radio_manual_state.attributes = {
            "powered_by": "switch.kitchen_main_power",
            "activation_mode": "manual",
        }
        
        # Mock the power switch state as OFF
        switch_state = MagicMock()
        switch_state.state = "off"
        
        # Update hass.states.get to return our mocks
        def mock_get(eid):
            if eid == "light.radio_manual":
                return radio_manual_state
            if eid == "switch.kitchen_main_power":
                return switch_state
            return None
            
        hass.states.get = mock_get
        
        usable, filtered = await capability.filter_candidates_by_usability(["light.radio_manual"])
        # If switch.kitchen_main_power is OFF, light.radio_manual should be filtered
        assert "light.radio_manual" in filtered
