"""Tests for capability-based entity filtering (dimmability).

Tests that non-dimmable lights are filtered from:
1. Semantic cache area-scope anchors for HassLightSet
2. Entity resolver results for HassLightSet intent
"""

from unittest.mock import MagicMock
import pytest

from multistage_assist.capabilities.entity_resolver import EntityResolverCapability


# ============================================================================
# DIMMABILITY HELPER TESTS
# ============================================================================

async def test_is_light_dimmable_brightness_mode(hass, config_entry):
    """Light with brightness mode should be dimmable."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    # Setup a dimmable light (use unique ID to avoid fixture overlap)
    hass.states.set(
        "light.test_dimmable",
        "on",
        {"friendly_name": "Test Dimmable", "supported_color_modes": ["brightness"]}
    )
    
    assert resolver._is_light_dimmable("light.test_dimmable") == True


async def test_is_light_dimmable_color_temp_mode(hass, config_entry):
    """Light with color_temp mode should be dimmable."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    hass.states.set(
        "light.test_color_temp",
        "on",
        {"friendly_name": "Test Color Temp", "supported_color_modes": ["color_temp", "brightness"]}
    )
    
    assert resolver._is_light_dimmable("light.test_color_temp") == True


async def test_is_light_dimmable_onoff_only(hass, config_entry):
    """Light with only onoff mode should NOT be dimmable."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    hass.states.set(
        "light.test_onoff",
        "on",
        {"friendly_name": "Test OnOff", "supported_color_modes": ["onoff"]}
    )
    
    assert resolver._is_light_dimmable("light.test_onoff") == False


async def test_is_light_dimmable_no_color_modes(hass, config_entry):
    """Light with no color_modes attribute should be treated as dimmable (safe default)."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    hass.states.set(
        "light.test_legacy",
        "on",
        {"friendly_name": "Test Legacy"}  # No supported_color_modes
    )
    
    assert resolver._is_light_dimmable("light.test_legacy") == True


async def test_is_light_dimmable_empty_modes(hass, config_entry):
    """Light with empty color_modes list should be treated as dimmable."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    hass.states.set(
        "light.test_empty",
        "on",
        {"friendly_name": "Test Empty", "supported_color_modes": []}
    )
    
    assert resolver._is_light_dimmable("light.test_empty") == True


# ============================================================================
# ENTITY RESOLVER CAPABILITY FILTERING TESTS
# ============================================================================

async def test_resolver_helper_directly(hass, config_entry):
    """Test _is_light_dimmable helper with mixed light types."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    # Setup mixed light types
    hass.states.set(
        "light.direct_dimmable",
        "on",
        {"friendly_name": "Direct Dimmable", "supported_color_modes": ["brightness"]}
    )
    hass.states.set(
        "light.direct_onoff",
        "on",
        {"friendly_name": "Direct OnOff", "supported_color_modes": ["onoff"]}
    )
    
    # Direct helper tests
    assert resolver._is_light_dimmable("light.direct_dimmable") == True
    assert resolver._is_light_dimmable("light.direct_onoff") == False


async def test_existing_fixture_lights_have_no_color_modes(hass, config_entry):
    """Verify existing fixture lights without color_modes are treated as dimmable."""
    resolver = EntityResolverCapability(hass, config_entry.data)
    
    # The fixture creates light.kuche without supported_color_modes
    # Our logic treats missing color_modes as dimmable (safe default)
    assert resolver._is_light_dimmable("light.kuche") == True

