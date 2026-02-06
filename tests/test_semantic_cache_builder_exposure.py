"""Test that unexposed entities are NOT included in anchor generation.

This is a regression test to ensure we never accidentally add cache entries
for entities that are not exposed to the Assist/Conversation component.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np


@pytest.mark.asyncio
async def test_unexposed_entities_are_excluded(hass):
    """Test that entities NOT exposed to conversation are excluded from anchors."""
    from multistage_assist.capabilities.semantic_cache_builder import SemanticCacheBuilder
    
    # 1. Setup Mock Registries
    # Mock Area Registry
    areg = MagicMock()
    kitchen = MagicMock()
    kitchen.id = "area_kitchen"
    kitchen.name = "Küche"
    kitchen.floor_id = None
    
    living = MagicMock()
    living.id = "area_living"
    living.name = "Wohnzimmer"
    living.floor_id = None
    
    areg.async_list_areas.return_value = [kitchen, living]
    
    # Mock Floor Registry
    freg = MagicMock()
    freg.async_list_floors.return_value = []
    
    # Mock Entity Registry
    ereg = MagicMock()
    entities = {}
    
    def create_entity(eid, original_name, area_id):
        e = MagicMock()
        e.entity_id = eid
        e.name = None
        e.original_name = original_name
        e.area_id = area_id
        e.disabled = False
        entities[eid] = e
        return e

    # EXPOSED Entity: "Wohnzimmer" (light) -> Should Generate
    create_entity("light.wohnzimmer", "Wohnzimmer Licht", living.id)
    
    # NOT EXPOSED Entity: "Gäste Badezimmer Heizung" (climate) -> Should Skip
    create_entity("climate.gastebad_heizung", "Gäste Badezimmer Heizung", None)
    
    # NOT EXPOSED Entity: "FRITZ!Box Port Forward" (switch) -> Should Skip
    create_entity("switch.fritzbox_port_forward", "FRITZ!Box 7520 (UI) Port forward Home Assistant", None)
    
    # EXPOSED Entity: "Küche" (light) -> Should Generate
    create_entity("light.kuche", "Küche Licht", kitchen.id)
    
    ereg.entities = entities

    # 2. Setup Builder with AsyncMock for embeddings
    mock_get_embedding = AsyncMock(return_value=np.array([0.1] * 768))
    
    builder = SemanticCacheBuilder(hass, {}, mock_get_embedding, MagicMock(side_effect=lambda x: (x, False)))
    
    # Define which entities are exposed
    exposed_entities = {"light.wohnzimmer", "light.kuche"}
    
    def mock_should_expose(hass, domain, entity_id):
        return entity_id in exposed_entities
    
    # 3. Patch Registries and async_should_expose
    with patch("homeassistant.helpers.area_registry.async_get", return_value=areg), \
         patch("homeassistant.helpers.floor_registry.async_get", return_value=freg), \
         patch("homeassistant.helpers.entity_registry.async_get", return_value=ereg), \
         patch("multistage_assist.capabilities.semantic_cache_builder.async_should_expose", mock_should_expose), \
         patch("multistage_assist.capabilities.keyword_intent.KeywordIntentCapability") as mock_ki:
        
        # Mock INTENT_DATA required for domain filtering
        mock_ki.INTENT_DATA = {
            "light": {},
            "switch": {},
            "cover": {},
            "climate": {},
            "fan": {},
            "media_player": {},
            "automation": {},
        }
        
        anchors = await builder.generate_anchors()
        
    anchor_texts = [a.text for a in anchors]
    anchor_entity_ids = set()
    for a in anchors:
        anchor_entity_ids.update(a.entity_ids)
    
    # 4. Assertions
    
    # EXPOSED: light.wohnzimmer and light.kuche should have anchors
    assert "light.wohnzimmer" in anchor_entity_ids, "Exposed entity light.wohnzimmer should generate anchors"
    assert "light.kuche" in anchor_entity_ids, "Exposed entity light.kuche should generate anchors"
    
    # NOT EXPOSED: These should NOT appear anywhere
    assert "climate.gastebad_heizung" not in anchor_entity_ids, \
        "Unexposed entity climate.gastebad_heizung should NOT generate anchors"
    assert "switch.fritzbox_port_forward" not in anchor_entity_ids, \
        "Unexposed entity switch.fritzbox_port_forward should NOT generate anchors"
    
    # Also check text patterns don't contain unexposed entity names
    for text in anchor_texts:
        assert "Gäste Badezimmer Heizung" not in text, f"Unexposed entity name in anchor text: {text}"
        assert "FRITZ!Box" not in text, f"Unexposed entity name in anchor text: {text}"


@pytest.mark.asyncio
async def test_exposure_check_failure_falls_back_open(hass):
    """Test that if async_should_expose fails, we still include entities (fail open for compatibility)."""
    from multistage_assist.capabilities.semantic_cache_builder import SemanticCacheBuilder
    
    # 1. Setup Mock Registries
    areg = MagicMock()
    kitchen = MagicMock()
    kitchen.id = "area_kitchen"
    kitchen.name = "Küche"
    kitchen.floor_id = None
    areg.async_list_areas.return_value = [kitchen]
    
    freg = MagicMock()
    freg.async_list_floors.return_value = []
    
    ereg = MagicMock()
    e = MagicMock()
    e.entity_id = "light.kuche"
    e.name = None
    e.original_name = "Küche Licht"
    e.area_id = kitchen.id
    e.disabled = False
    ereg.entities = {"light.kuche": e}

    mock_get_embedding = AsyncMock(return_value=np.array([0.1] * 768))
    builder = SemanticCacheBuilder(hass, {}, mock_get_embedding, MagicMock(side_effect=lambda x: (x, False)))
    
    def mock_should_expose_fails(hass, domain, entity_id):
        raise RuntimeError("Exposure check unavailable")
    
    # Patch to simulate exposure check failure
    with patch("homeassistant.helpers.area_registry.async_get", return_value=areg), \
         patch("homeassistant.helpers.floor_registry.async_get", return_value=freg), \
         patch("homeassistant.helpers.entity_registry.async_get", return_value=ereg), \
         patch("multistage_assist.capabilities.semantic_cache_builder.async_should_expose", mock_should_expose_fails), \
         patch("multistage_assist.capabilities.keyword_intent.KeywordIntentCapability") as mock_ki:
        
        mock_ki.INTENT_DATA = {
            "light": {},
            "switch": {},
            "cover": {},
            "climate": {},
            "fan": {},
            "media_player": {},
            "automation": {},
        }
        
        anchors = await builder.generate_anchors()
        
    anchor_entity_ids = set()
    for a in anchors:
        anchor_entity_ids.update(a.entity_ids)
    
    # When exposure check fails, entity should still be included (fail open)
    assert "light.kuche" in anchor_entity_ids, \
        "Entity should be included when exposure check fails (fail open for compatibility)"
