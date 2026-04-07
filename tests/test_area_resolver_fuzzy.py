"""Tests for Area and Floor fuzzy resolution."""

import pytest
from unittest.mock import MagicMock
from multistage_assist.capabilities.area_resolver import AreaResolverCapability

class MockArea:
    def __init__(self, name, aliases=None):
        self.name = name
        self.aliases = aliases or set()
        
class MockRegistry:
    def __init__(self, items):
        self.items = items
    def async_list_areas(self):
        return self.items
    def async_list_floors(self):
        return self.items

def test_area_fuzzy_matching(monkeypatch):
    """Test that typos in area names correctly resolve using fuzzy matching."""
    hass = MagicMock()
    resolver = AreaResolverCapability(hass, {})
    
    # Mock area_registry returned areas
    areas = [
        MockArea("Garage", {"Auto"}),
        MockArea("Badezimmer", {"Bad"}),
        MockArea("Schlafzimmer"),
    ]
    
    import homeassistant.helpers.area_registry as ar
    monkeypatch.setattr(ar, "async_get", MagicMock(return_value=MockRegistry(areas)))
    
    # "Gaeage" has a typo, should match "Garage"
    res = resolver.find_area("Gaeage")
    assert res is not None
    assert res.name == "Garage"
    
    # "Bqdezimmer" typo
    res2 = resolver.find_area("Bqdezimmer")
    assert res2 is not None
    assert res2.name == "Badezimmer"
    
    # Unrelated name should not match
    res3 = resolver.find_area("Wohnzimmer")
    assert res3 is None

def test_area_resolver_safety_guards(monkeypatch):
    """Test safety guards for partial matches."""
    hass = MagicMock()
    resolver = AreaResolverCapability(hass, {})
    
    # Mock area_registry returned areas
    areas = [
        MockArea("Gäste Badezimmer", {"Gästebad"}),
        MockArea("Badezimmer"),
        MockArea("Kinder Badezimmer"),
    ]
    
    import homeassistant.helpers.area_registry as ar
    monkeypatch.setattr(ar, "async_get", MagicMock(return_value=MockRegistry(areas)))

    # "Bade" (4) in "Badezimmer" (10) -> ratio 0.4 (fail)
    res = resolver.find_area("Bade")
    assert res is None  # Should fail guard!

    # "Badez" (5) in "Badezimmer" (10) -> ratio 0.5 (pass)
    res2 = resolver.find_area("Badez")
    assert res2 is not None
    assert res2.name == "Badezimmer"

def test_floor_resolver_safety_guards(monkeypatch):
    """Test safety guards for floor partial matches."""
    hass = MagicMock()
    resolver = AreaResolverCapability(hass, {})
    
    class MockFloor:
        def __init__(self, name, floor_id):
            self.name = name
            self.floor_id = floor_id
    
    # Mock floor_registry returned floors
    floors = [
        MockFloor("Erdgeschoss", floor_id="eg"),
        MockFloor("Obergeschoss", floor_id="og"),
        MockFloor("Untergeschoss", floor_id="ug"),
    ]
    
    import homeassistant.helpers.floor_registry as fr
    monkeypatch.setattr(fr, "async_get", MagicMock(return_value=MockRegistry(floors)))

    # "Erdge" (5) in "Erdgeschoss" (12) -> ratio 0.41 (fail)
    res = resolver.find_floor("Erdge")
    assert res is None
    
    # "Erdgesch" (8) in "Erdgeschoss" (12) -> ratio 0.66 (pass)
    res2 = resolver.find_floor("Erdgesch")
    assert res2 is not None
    assert res2.name == "Erdgeschoss"
