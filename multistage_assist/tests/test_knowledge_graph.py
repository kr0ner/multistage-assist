"""Tests for the Knowledge Graph module."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from multistage_assist.utils.knowledge_graph import (
    KnowledgeGraph,
    RelationType,
    ActivationMode,
    DependencyResolution,
    get_knowledge_graph,
    resolve_dependencies,
    is_entity_usable,
    filter_usable_entities,
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
    
    # Ambilight coupled with TV
    ambilight_state = MagicMock()
    ambilight_state.entity_id = "light.living_room_ambilight"
    ambilight_state.state = "off"
    ambilight_state.attributes = {
        "friendly_name": "Ambilight",
        "coupled_with": "media_player.living_room_tv",
        "activation_mode": "sync",
    }
    states["light.living_room_ambilight"] = ambilight_state
    
    # TV (on)
    tv_state = MagicMock()
    tv_state.entity_id = "media_player.living_room_tv"
    tv_state.state = "playing"
    tv_state.attributes = {"friendly_name": "Living Room TV"}
    states["media_player.living_room_tv"] = tv_state
    
    # Light with lux sensor
    light_state = MagicMock()
    light_state.entity_id = "light.living_room_main"
    light_state.state = "off"
    light_state.attributes = {
        "friendly_name": "Living Room Light",
        "lux_sensor": "sensor.living_room_lux",
        "min_lux_threshold": 200,
        "associated_cover": "cover.living_room_blinds",
    }
    states["light.living_room_main"] = light_state
    
    # Lux sensor (bright)
    lux_state = MagicMock()
    lux_state.entity_id = "sensor.living_room_lux"
    lux_state.state = "350"
    lux_state.attributes = {"friendly_name": "Living Room Lux"}
    states["sensor.living_room_lux"] = lux_state
    
    # Cover (closed)
    cover_state = MagicMock()
    cover_state.entity_id = "cover.living_room_blinds"
    cover_state.state = "closed"
    cover_state.attributes = {"friendly_name": "Living Room Blinds"}
    states["cover.living_room_blinds"] = cover_state
    
    # Simple light without dependencies
    simple_light = MagicMock()
    simple_light.entity_id = "light.bedroom"
    simple_light.state = "off"
    simple_light.attributes = {"friendly_name": "Bedroom Light"}
    states["light.bedroom"] = simple_light
    
    # Set up hass.states
    def get_state(entity_id):
        return states.get(entity_id)
    
    hass.states.get = get_state
    hass.states.async_all = MagicMock(return_value=list(states.values()))
    
    return hass


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""
    
    def test_refresh_graph_loads_dependencies(self, hass):
        """Test that refresh_graph loads all dependencies."""
        graph = KnowledgeGraph(hass)
        graph.refresh_graph()
        
        # Should have 3 entities with dependencies
        assert len(graph._dependencies) == 3
        
        # Kitchen radio should have power dependency
        radio_deps = graph.get_dependencies("media_player.kitchen_radio")
        assert len(radio_deps) == 1
        assert radio_deps[0].relation_type == RelationType.POWERED_BY
        assert radio_deps[0].target_entity == "switch.kitchen_main_power"
    
    def test_get_power_dependency(self, hass):
        """Test getting power dependency for an entity."""
        graph = KnowledgeGraph(hass)
        
        power_source = graph.get_power_dependency("media_player.kitchen_radio")
        assert power_source == "switch.kitchen_main_power"
        
        # Entity without power dependency
        no_power = graph.get_power_dependency("light.bedroom")
        assert no_power is None
    
    def test_get_coupled_device(self, hass):
        """Test getting coupled device for an entity."""
        graph = KnowledgeGraph(hass)
        
        coupled = graph.get_coupled_device("light.living_room_ambilight")
        assert coupled == "media_player.living_room_tv"
    
    def test_is_entity_usable_power_off(self, hass):
        """Test that entity with power off is not usable."""
        graph = KnowledgeGraph(hass)
        
        usable, reason = graph.is_entity_usable("media_player.kitchen_radio")
        assert usable is False
        assert "kitchen_main_power" in reason
    
    def test_is_entity_usable_coupled_active(self, hass):
        """Test that coupled entity is usable when coupled device is on."""
        graph = KnowledgeGraph(hass)
        
        # TV is playing, so ambilight is usable
        usable, reason = graph.is_entity_usable("light.living_room_ambilight")
        assert usable is True
    
    def test_is_entity_usable_no_dependencies(self, hass):
        """Test that entity without dependencies is always usable."""
        graph = KnowledgeGraph(hass)
        
        usable, reason = graph.is_entity_usable("light.bedroom")
        assert usable is True
        assert reason is None


class TestDependencyResolution:
    """Tests for dependency resolution."""
    
    def test_resolve_power_dependency_auto(self, hass):
        """Test resolving power dependency with auto mode."""
        graph = KnowledgeGraph(hass)
        
        result = graph.resolve_for_action("media_player.kitchen_radio", "turn_on")
        
        assert result.can_proceed is True
        assert len(result.prerequisites) == 1
        assert result.prerequisites[0]["entity_id"] == "switch.kitchen_main_power"
        assert result.prerequisites[0]["action"] == "turn_on"
    
    def test_resolve_coupled_device_sync(self, hass):
        """Test resolving coupled device with sync mode."""
        # Change TV state to off
        hass.states.get("media_player.living_room_tv").state = "off"
        
        graph = KnowledgeGraph(hass)
        
        result = graph.resolve_for_action("light.living_room_ambilight", "turn_on")
        
        # With sync mode, should turn on TV first
        assert len(result.prerequisites) == 1
        assert result.prerequisites[0]["entity_id"] == "media_player.living_room_tv"
    
    def test_resolve_lux_suggestion_bright(self, hass):
        """Test that bright room suggests not turning on light."""
        graph = KnowledgeGraph(hass)
        
        result = graph.resolve_for_action("light.living_room_main", "turn_on")
        
        # Should have a suggestion about brightness
        assert len(result.suggestions) >= 1
        assert any("hell" in s.lower() or "lux" in s.lower() for s in result.suggestions)
    
    def test_resolve_cover_suggestion(self, hass):
        """Test that closed cover suggests opening instead of light."""
        # Set lux to indicate outdoor brightness
        hass.states.get("sensor.living_room_lux").state = "500"
        
        graph = KnowledgeGraph(hass)
        
        result = graph.resolve_for_action("light.living_room_main", "turn_on")
        
        # Should suggest opening blinds
        assert any("rollo" in s.lower() or "öffnen" in s.lower() for s in result.suggestions)
    
    def test_resolve_no_dependencies(self, hass):
        """Test resolving entity without dependencies."""
        graph = KnowledgeGraph(hass)
        
        result = graph.resolve_for_action("light.bedroom", "turn_on")
        
        assert result.can_proceed is True
        assert len(result.prerequisites) == 0
        assert len(result.warnings) == 0


class TestFilterCandidates:
    """Tests for filtering entity candidates."""
    
    def test_filter_usable_entities(self, hass):
        """Test filtering entities by usability."""
        graph = KnowledgeGraph(hass)
        
        candidates = [
            "media_player.kitchen_radio",  # Not usable (power off)
            "light.living_room_ambilight",  # Usable (TV is on)
            "light.bedroom",  # Usable (no deps)
        ]
        
        usable, filtered = graph.filter_candidates_by_usability(candidates)
        
        assert "media_player.kitchen_radio" in filtered
        assert "light.living_room_ambilight" in usable
        assert "light.bedroom" in usable


class TestModuleFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_knowledge_graph_caching(self, hass):
        """Test that knowledge graph is cached per hass instance."""
        graph1 = get_knowledge_graph(hass)
        graph2 = get_knowledge_graph(hass)
        
        assert graph1 is graph2
    
    def test_resolve_dependencies_shortcut(self, hass):
        """Test the resolve_dependencies shortcut function."""
        result = resolve_dependencies(hass, "light.bedroom", "turn_on")
        
        assert isinstance(result, DependencyResolution)
        assert result.can_proceed is True
    
    def test_is_entity_usable_shortcut(self, hass):
        """Test the is_entity_usable shortcut function."""
        usable, reason = is_entity_usable(hass, "light.bedroom")
        
        assert usable is True
    
    def test_filter_usable_entities_shortcut(self, hass):
        """Test the filter_usable_entities shortcut function."""
        candidates = ["light.bedroom", "media_player.kitchen_radio"]
        
        usable = filter_usable_entities(hass, candidates)
        
        assert "light.bedroom" in usable
        assert "media_player.kitchen_radio" not in usable


class TestGraphSummary:
    """Tests for graph debugging/summary."""
    
    def test_get_graph_summary(self, hass):
        """Test getting graph summary."""
        graph = KnowledgeGraph(hass)
        summary = graph.get_graph_summary()
        
        assert "total_entities_with_deps" in summary
        assert "total_dependencies" in summary
        assert "by_type" in summary
        assert summary["total_entities_with_deps"] > 0
