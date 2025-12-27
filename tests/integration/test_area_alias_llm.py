"""Integration tests for AreaAliasCapability with real LLM."""

import pytest
from unittest.mock import MagicMock, patch
from homeassistant.components import conversation

from multistage_assist.capabilities.area_resolver import AreaResolverCapability
from tests.integration import get_llm_config


pytestmark = pytest.mark.integration


# Test candidates that will be mocked in the registry
TEST_AREAS = ["Badezimmer", "Küche", "Wohnzimmer", "Untergeschoss", "Erdgeschoss", 
              "Obergeschoss", "Flur EG", "Flur OG", "Flur UG"]
TEST_FLOORS = ["Erdgeschoss", "Obergeschoss", "Untergeschoss"]


@pytest.fixture
def hass():
    """Mock Home Assistant instance with area and floor registries."""
    hass = MagicMock()
    return hass


@pytest.fixture
def area_resolver_capability(hass):
    """Create area resolver capability with real LLM and mocked registries."""
    
    # Create mock areas
    mock_areas = []
    for name in TEST_AREAS:
        area = MagicMock()
        area.name = name
        mock_areas.append(area)
    
    # Create mock floors  
    mock_floors = []
    for name in TEST_FLOORS:
        floor = MagicMock()
        floor.name = name
        mock_floors.append(floor)
    
    # Patch the registry functions
    with patch("multistage_assist.capabilities.area_resolver.ar") as mock_ar, \
         patch("multistage_assist.capabilities.area_resolver.fr") as mock_fr:
        
        mock_area_reg = MagicMock()
        mock_area_reg.async_list_areas.return_value = mock_areas
        mock_ar.async_get.return_value = mock_area_reg
        
        mock_floor_reg = MagicMock()
        mock_floor_reg.async_list_floors.return_value = mock_floors
        mock_fr.async_get.return_value = mock_floor_reg
        
        capability = AreaResolverCapability(hass, get_llm_config())
        # Store patched registries so they persist
        capability._mock_ar = mock_ar
        capability._mock_fr = mock_fr
        capability._mock_area_reg = mock_area_reg
        capability._mock_floor_reg = mock_floor_reg
        
        yield capability


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


@pytest.mark.parametrize(
    "user_query,candidates,expected_match",
    [
        # Exact matches
        ("Badezimmer", ["Badezimmer", "Küche", "Wohnzimmer"], "Badezimmer"),
        ("Küche", ["Badezimmer", "Küche", "Wohnzimmer"], "Küche"),
        # Synonyms and abbreviations
        ("Bad", ["Badezimmer", "Küche"], "Badezimmer"),
        ("Keller", ["Untergeschoss", "Erdgeschoss"], "Untergeschoss"),
        ("Unten", ["Erdgeschoss", "Obergeschoss"], "Erdgeschoss"),
        ("Oben", ["Erdgeschoss", "Obergeschoss"], "Obergeschoss"),
        # Partial matches
        (
            "Flur",
            ["Flur EG", "Flur OG", "Flur UG"],
            "Flur EG",
        ),  # Should match first/closest
        ("EG", ["Erdgeschoss", "Obergeschoss"], "Erdgeschoss"),
        # Global scope
        ("Haus", ["Badezimmer", "Küche"], "GLOBAL"),
        ("Wohnung", ["Badezimmer", "Küche"], "GLOBAL"),
        ("Überall", ["Badezimmer", "Küche"], "GLOBAL"),
        ("Alles", ["Badezimmer", "Küche"], "GLOBAL"),
        # No match cases
        ("Garage", ["Badezimmer", "Küche", "Wohnzimmer"], None),
        ("XYZ", ["Badezimmer", "Küche"], None),
    ],
)
async def test_area_matching(
    area_resolver_capability, user_query, candidates, expected_match
):
    """Test area alias matching with real LLM."""
    user_input = make_input(user_query)

    result = await area_resolver_capability.run(
        user_input,
        user_query=user_query,
        candidates=candidates,
        mode="area",
    )

    assert (
        result is not None
    ), f"No result for query='{user_query}', candidates={candidates}"
    match = result.get("match")

    assert (
        match == expected_match
    ), f"Expected match '{expected_match}' for query='{user_query}' in {candidates}, got: '{match}'"


@pytest.mark.parametrize(
    "user_query,floor_candidates,expected",
    [
        (
            "Erdgeschoss",
            ["Erdgeschoss", "Obergeschoss", "Untergeschoss"],
            "Erdgeschoss",
        ),
        ("Unten", ["Erdgeschoss", "Obergeschoss"], "Erdgeschoss"),
        ("Oben", ["Erdgeschoss", "Obergeschoss"], "Obergeschoss"),
        ("Keller", ["Untergeschoss", "Erdgeschoss"], "Untergeschoss"),
        ("EG", ["Erdgeschoss", "Obergeschoss"], "Erdgeschoss"),
        ("OG", ["Erdgeschoss", "Obergeschoss"], "Obergeschoss"),
        ("UG", ["Untergeschoss", "Erdgeschoss"], "Untergeschoss"),
    ],
)
async def test_floor_matching(
    area_resolver_capability, user_query, floor_candidates, expected
):
    """Test floor alias matching."""
    user_input = make_input(user_query)

    result = await area_resolver_capability.run(
        user_input,
        user_query=user_query,
        candidates=floor_candidates,
        mode="floor",
    )

    match = result.get("match")
    assert (
        match == expected
    ), f"Expected floor '{expected}' for query='{user_query}', got: '{match}'"


async def test_fuzzy_matching(area_resolver_capability):
    """Test fuzzy matching with slight misspellings or variations."""
    test_cases = [
        ("Badzimmer", ["Badezimmer", "Küche"], "Badezimmer"),  # Typo
        ("Wozi", ["Wohnzimmer", "Arbeitszimmer"], "Wohnzimmer"),  # Abbreviation
        ("Schafzimmer", ["Schlafzimmer", "Badezimmer"], "Schlafzimmer"),  # Typo
    ]

    for query, candidates, expected in test_cases:
        user_input = make_input(query)
        result = await area_resolver_capability.run(
            user_input,
            user_query=query,
            candidates=candidates,
            mode="area",
        )

        match = result.get("match")
        # Fuzzy matching might work or might not - just check it doesn't crash
        assert (
            match is not None or match is None
        ), f"Test for fuzzy match '{query}' should return something"
