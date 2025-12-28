import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import floor_registry as fr
from multistage_assist.capabilities.semantic_cache_builder import SemanticCacheBuilder

@pytest.mark.asyncio
async def test_generate_unique_entity_anchors(hass):
    """Test generating anchors for globally unique entities."""
    
    # 1. Setup Mock Registries
    # Mock Area Registry
    areg = MagicMock()
    kitchen = MagicMock()
    kitchen.id = "area_kitchen"
    kitchen.name = "Kitchen"
    kitchen.floor_id = None
    
    living = MagicMock()
    living.id = "area_living"
    living.name = "Living Room"
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

    # Unique Name: "Ambilight" (light) -> Should Generate
    create_entity("light.hue_1", "Ambilight", living.id)
    
    # Duplicate Name: "Licht" (light) -> Should Skip
    create_entity("light.hue_2", "Licht", kitchen.id)
    create_entity("light.hue_3", "Licht", living.id)
    
    # Name Conflict: "Kitchen" (switch) matches Area -> Should Skip
    create_entity("switch.tplink_4", "Kitchen", kitchen.id)
    
    ereg.entities = entities

    # 2. Setup Builder with AsyncMock for embeddings
    # Mock get_embedding (must be async)
    mock_get_embedding = AsyncMock(return_value=np.array([0.1] * 768))
    
    builder = SemanticCacheBuilder(hass, {}, mock_get_embedding, MagicMock(side_effect=lambda x: (x, False)))
    
    # 3. Patch Registries and KeywordIntentCapability
    with patch("homeassistant.helpers.area_registry.async_get", return_value=areg), \
         patch("homeassistant.helpers.floor_registry.async_get", return_value=freg), \
         patch("homeassistant.helpers.entity_registry.async_get", return_value=ereg), \
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
    
    # 4. Assertions
    
    # Unique Entity: "Ambilight"
    # Expect: "Schalte das Ambilight an"
    assert "Schalte das Ambilight an" in anchor_texts
    
    # Duplicate Entity: "Licht"
    # Should NOT generate global unique anchor "Schalte das Licht an"
    assert "Schalte das Licht an" not in anchor_texts
    
    # Conflict Entity: "Kitchen"
    # Should NOT generate "Schalte der Kitchen an" (switch)
    assert "Schalte der Kitchen an" not in anchor_texts
