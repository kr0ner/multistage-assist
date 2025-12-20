"""Integration tests for IntentConfirmationCapability with real LLM."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.intent_confirmation import (
    IntentConfirmationCapability,
)
from tests.integration import get_llm_config


pytestmark = pytest.mark.integration


@pytest.fixture
def hass():
    """Mock Home Assistant instance with state."""
    hass = MagicMock()

    # Mock state objects
    def get_state(entity_id):
        state_obj = MagicMock()
        state_obj.attributes = {
            "friendly_name": entity_id.split(".")[-1].replace("_", " ").title()
        }
        state_obj.state = "off"
        return state_obj

    hass.states.get = get_state
    return hass


@pytest.fixture
def confirmation_capability(hass):
    """Create intent confirmation capability with real LLM."""
    return IntentConfirmationCapability(hass, get_llm_config())


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
    "intent,devices,params,expected_keywords",
    [
        # Turn on
        ("HassTurnOn", ["Küche"], {}, ["küche", "an", "ist"]),
        (
            "HassTurnOn",
            ["Küche", "Wohnzimmer"],
            {},
            ["küche", "wohnzimmer", "an"],  # LLM may use separate "ist" sentences or plural "sind"
        ),
        # Turn off
        ("HassTurnOff", ["Badezimmer"], {}, ["badezimmer", "aus"]),
        ("HassTurnOff", ["Flur", "Büro"], {}, ["flur", "büro", "aus"]),
        # Light set (brightness)
        (
            "HassLightSet",
            ["Schlafzimmer"],
            {"brightness": "50"},
            ["schlafzimmer"],  # LLM may describe as "dunkler" rather than "50"
        ),
        ("HassLightSet", ["Küche"], {"brightness": "step_up"}, ["küche", "heller"]),
        # Position - accept Rollo OR Rollladen
        ("HassSetPosition", ["Rollo Bad"], {"position": "75"}, ["75"]),  # rollo/rollladen checked separately
        # Temporary control with duration
        (
            "TemporaryControl",
            ["Flur"],
            {"duration": "10 Minuten", "command": "on"},
            ["flur", "10", "minuten"],
        ),
        (
            "TemporaryControl",
            ["Garage"],
            {"duration": "5 Minuten", "command": "off"},
            ["garage", "5", "minuten"],
        ),
        # Climate
        (
            "HassClimateSetTemperature",
            ["Wohnzimmer"],
            {"temperature": "22"},
            ["wohnzimmer", "22"],
        ),
        # Multiple devices
        ("HassTurnOn", ["Licht 1", "Licht 2", "Licht 3"], {}, ["licht"]),
    ],
)
async def test_confirmation_generation(
    confirmation_capability, intent, devices, params, expected_keywords
):
    """Test confirmation message generation with real LLM."""
    user_input = make_input("test")

    # Mock entity IDs
    entity_ids = [f"light.{d.lower().replace(' ', '_')}" for d in devices]

    result = await confirmation_capability.run(
        user_input,
        intent_name=intent,
        entity_ids=entity_ids,
        params=params,
    )

    assert result is not None, f"No confirmation for intent={intent}, devices={devices}"
    # Result is a dict with 'message' key
    message = result.get("message", "")
    assert message, f"Empty message for intent={intent}"

    response_text = message.lower()

    # Check that expected keywords are in response
    for keyword in expected_keywords:
        assert (
            keyword in response_text
        ), f"Expected keyword '{keyword}' in response for {intent}/{devices}, got: {message}"
    
    # For cover/position intents, accept Rollo OR Rollladen
    if intent == "HassSetPosition" and "Rollo" in str(devices):
        assert (
            "rollo" in response_text or "rollladen" in response_text
        ), f"Expected 'rollo' or 'rollladen' in response for {intent}/{devices}, got: {message}"


@pytest.mark.parametrize(
    "device_count,expected_verb",
    [
        (1, "ist"),  # Singular
        (2, "sind"),  # Plural (or multiple "ist")
        (3, "sind"),  # Plural (or multiple "ist")
    ],
)
async def test_verb_conjugation(confirmation_capability, device_count, expected_verb):
    """Test correct German verb conjugation (ist vs sind)."""
    user_input = make_input("test")
    devices = [f"Gerät {i}" for i in range(device_count)]
    entity_ids = [f"light.gerat_{i}" for i in range(device_count)]

    result = await confirmation_capability.run(
        user_input,
        intent_name="HassTurnOn",
        entity_ids=entity_ids,
        params={},
    )

    assert result is not None
    message = result.get("message", "")
    response_text = message.lower()
    
    # Accept either:
    # 1. Plural "sind" for multiple devices
    # 2. Multiple "ist" sentences (one per device) - also valid
    ist_count = response_text.count("ist")
    has_valid_conjugation = (
        expected_verb in response_text or 
        (device_count > 1 and ist_count >= device_count)  # Multiple "ist" for each device
    )
    assert has_valid_conjugation, f"Expected verb '{expected_verb}' or {device_count}x 'ist' for {device_count} devices, got: {message}"


async def test_duration_in_confirmation(confirmation_capability):
    """Test that duration is mentioned in temporary control confirmations."""
    user_input = make_input("test")

    duration_tests = [
        ("10 Minuten", ["10", "minuten"]),
        ("5 Sekunden", ["5", "sekunden"]),
        ("1 Stunde", ["stunde"]),
    ]

    for duration, keywords in duration_tests:
        result = await confirmation_capability.run(
            user_input,
            intent_name="TemporaryControl",
            entity_ids=["light.test"],
            params={"duration": duration, "command": "on"},
        )

        assert result is not None
        message = result.get("message", "")
        response_lower = message.lower()

        for keyword in keywords:
            assert (
                keyword in response_lower
            ), f"Expected '{keyword}' in confirmation for duration '{duration}', got: {message}"


async def test_present_tense_usage(confirmation_capability):
    """Test that confirmations use present tense (ist/sind), not past tense."""
    user_input = make_input("test")

    result = await confirmation_capability.run(
        user_input,
        intent_name="HassTurnOn",
        entity_ids=["light.kuche"],
        params={},
    )

    assert result is not None
    message = result.get("message", "")
    response_lower = message.lower()

    # Should use "ist an" not "wurde angeschaltet"
    assert "ist" in response_lower or "sind" in response_lower
    assert "wurde" not in response_lower
    assert "wurden" not in response_lower
