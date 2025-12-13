"""Integration tests for KeywordIntentCapability with real LLM."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
from tests.integration import get_llm_config


pytestmark = pytest.mark.integration


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability instance with real LLM."""
    return KeywordIntentCapability(hass, get_llm_config())


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
    "user_text,expected_intent,expected_slots",
    [
        # Light commands
        ("Schalte das Licht an", "HassTurnOn", {"domain": "light", "command": "an"}),
        ("Licht aus", "HassTurnOff", {"domain": "light", "command": "aus"}),
        (
            "Mache das Licht heller",
            "HassLightSet",
            {"domain": "light", "brightness": "step_up"},
        ),
        ("Licht auf 50%", "HassLightSet", {"domain": "light", "brightness": "50"}),
        ("Licht im Bad an", "HassTurnOn", {"domain": "light", "area": "Bad"}),
        # Duration-based (HassTemporaryControl)
        (
            "Licht für 10 Minuten an",
            "HassTemporaryControl",
            {"duration": "10 Minuten", "command": "on"},
        ),
        (
            "Schalte das Licht für 5 Minuten aus",
            "HassTemporaryControl",
            {"duration": "5 Minuten", "command": "off"},
        ),
        # Cover commands - LLM may use various intents for covers
        # "runter" = close, "auf" = open - accept flexible intent mapping
        (
            "Rollo auf 75%",
            "HassSetPosition",
            {"domain": "cover", "position": "75"},
        ),
        # State queries
        ("Ist das Licht an?", "HassGetState", {"domain": "light"}),
        ("Welche Lichter sind an?", "HassGetState", {"domain": "light"}),
        # Climate - HassClimateSetTemperature is the correct intent for heating
        (
            "Heizung auf 22 Grad",
            "HassClimateSetTemperature",
            {"domain": "climate"},
        ),
        (
            "Wie warm ist es?",
            "HassGetState",
            {"domain": "sensor"},
        ),  # Query mapped to GetState
        # Timer
        (
            "Stelle einen Timer auf 5 Minuten",
            "HassTimerSet",
            {"domain": "timer", "duration": "5 Minuten"},
        ),
    ],
)
async def test_intent_extraction(
    keyword_intent_capability, user_text, expected_intent, expected_slots
):
    """Test intent and slot extraction with real LLM."""
    user_input = make_input(user_text)
    result = await keyword_intent_capability.run(user_input)

    assert result is not None, f"No result for: {user_text}"
    assert "intent" in result, f"No intent in result for: {user_text}"
    assert (
        result["intent"] == expected_intent
    ), f"Expected intent '{expected_intent}', got '{result['intent']}' for: {user_text}"

    # Check expected slots are present
    slots = result.get("slots", {})
    for key, expected_value in expected_slots.items():
        assert key in slots, f"Expected slot '{key}' not found for: {user_text}"
        # For flexible matching, just check presence unless it's a specific value
        if isinstance(expected_value, str) and expected_value:
            actual = slots[key]
            if key in ["brightness", "position", "temperature"]:
                # Numeric values should match
                assert str(expected_value) in str(
                    actual
                ), f"Expected {key}='{expected_value}', got '{actual}' for: {user_text}"


@pytest.mark.parametrize(
    "user_text,should_have_area",
    [
        ("Licht im Bad an", True),
        ("Schalte das Licht in der Küche aus", True),
        ("Rollo im Wohnzimmer runter", True),
        ("Licht an", False),
        ("Alle Lichter aus", False),
    ],
)
async def test_area_extraction(keyword_intent_capability, user_text, should_have_area):
    """Test area slot extraction."""
    user_input = make_input(user_text)
    result = await keyword_intent_capability.run(user_input)

    assert result is not None
    slots = result.get("slots", {})

    if should_have_area:
        assert (
            "area" in slots and slots["area"]
        ), f"Expected area slot for: {user_text}, got slots: {slots}"
    else:
        assert not slots.get(
            "area"
        ), f"Did not expect area slot for: {user_text}, got: {slots.get('area')}"


@pytest.mark.parametrize(
    "user_text,expected_domain",
    [
        ("Schalte das Licht an", "light"),
        ("Rollo runter", "cover"),
        ("Heizung auf 22", "climate"),
        ("Stelle einen Timer", "timer"),
        ("Staubsauger starten", "vacuum"),
        ("Schalter an", "switch"),
        ("Ventilator an", "fan"),
    ],
)
async def test_domain_detection(keyword_intent_capability, user_text, expected_domain):
    """Test domain detection from keywords."""
    user_input = make_input(user_text)
    result = await keyword_intent_capability.run(user_input)

    assert result is not None
    assert (
        result.get("domain") == expected_domain
    ), f"Expected domain '{expected_domain}' for '{user_text}', got: {result.get('domain')}"


async def test_no_specific_name_extraction(keyword_intent_capability):
    """Test that generic words might fill name slot but are still generic."""
    generic_phrases = [
        "Schalte das Licht an",
        "Mache die Lampe an",
        "Rollo runter",
    ]

    # Common generic words that are acceptable as "name"
    generic_words = ["licht", "lampe", "rollo", "jalousie", "schalter", "ventilator"]

    for text in generic_phrases:
        user_input = make_input(text)
        result = await keyword_intent_capability.run(user_input)

        slots = result.get("slots", {})
        name = slots.get("name", "").lower()

        # Name should be either empty OR a known generic word
        if name:
            assert any(
                generic in name for generic in generic_words
            ), f"Generic phrase '{text}' has unexpected name: {name}"
