"""Tests for yes/no response capability."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.yes_no_response import YesNoResponseCapability


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()

    # Mock state objects
    def get_state(entity_id):
        state_obj = MagicMock()
        state_obj.attributes = {
            "friendly_name": entity_id.split(".")[-1].replace("_", " ").title()
        }
        return state_obj

    hass.states.get = get_state
    return hass


@pytest.fixture
def yes_no_capability(hass):
    """Create yes/no response capability instance."""
    config = {}
    return YesNoResponseCapability(hass, config)


@pytest.mark.parametrize(
    "text,expected_is_yes_no",
    [
        # YES/NO questions - should be detected
        ("Ist ein Licht an?", True),
        ("Sind Lichter an?", True),
        ("Gibt es offene Fenster?", True),
        ("Hat das Licht einen Timer?", True),
        ("Haben wir Rollläden offen?", True),
        ("Kann ich die Heizung einschalten?", True),
        ("Soll das Licht an bleiben?", True),
        # NOT yes/no questions - should NOT be detected
        ("Welche Lichter sind an?", False),
        ("Schalte das Licht an", False),
        ("Alle Lichter ausschalten", False),
    ],
)
async def test_yes_no_detection(yes_no_capability, text, expected_is_yes_no):
    """Test detection of yes/no questions."""
    user_input = conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )

    result = await yes_no_capability.run(
        user_input,
        domain="light",
        state="on",
        entity_ids=["light.kuche"],
    )

    if expected_is_yes_no:
        assert result is not None, f"Expected yes/no detection for: {text}"
        assert "Ja" in result or "Nein" in result
    else:
        assert result is None, f"Should not detect yes/no for: {text}"


@pytest.mark.parametrize(
    "entity_ids,expected_response",
    [
        # No matches
        ([], "Nein, kein Licht ist on."),
        # Single match
        (["light.kuche"], "Ja, Kuche ist on."),
        # Two matches
        (["light.kuche", "light.wohnzimmer"], "Ja, Kuche und Wohnzimmer sind on."),
        # Three matches
        (
            ["light.kuche", "light.wohnzimmer", "light.flur"],
            "Ja, Kuche, Wohnzimmer und Flur sind on.",
        ),
    ],
)
async def test_yes_no_response_formatting(
    yes_no_capability, entity_ids, expected_response
):
    """Test response formatting with different entity counts."""
    user_input = conversation.ConversationInput(
        text="Ist ein Licht an?",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )

    result = await yes_no_capability.run(
        user_input,
        domain="light",
        state="on",
        entity_ids=entity_ids,
    )

    assert result == expected_response


@pytest.mark.parametrize(
    "domain,expected_word",
    [
        ("light", "Licht"),
        ("cover", "Rollladen"),
        ("switch", "Schalter"),
        ("fan", "Ventilator"),
        ("climate", "Thermostat"),
        ("unknown_domain", "Gerät"),  # Fallback
    ],
)
async def test_domain_word_mapping(yes_no_capability, domain, expected_word):
    """Test correct German domain names."""
    user_input = conversation.ConversationInput(
        text="Ist etwas an?",
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )

    # No matches - should use domain word in response
    result = await yes_no_capability.run(
        user_input,
        domain=domain,
        state="on",
        entity_ids=[],
    )

    assert expected_word in result


def test_format_list_helper(yes_no_capability):
    """Test list formatting helper."""
    assert yes_no_capability._format_list(["A"]) == "A"
    assert yes_no_capability._format_list(["A", "B"]) == "A und B"
    assert yes_no_capability._format_list(["A", "B", "C"]) == "A, B und C"
    assert yes_no_capability._format_list(["A", "B", "C", "D"]) == "A, B, C und D"


def test_fast_detection(yes_no_capability):
    """Test keyword-based fast detection."""
    assert yes_no_capability._detect_yes_no_fast("ist ein licht an?") is True
    assert yes_no_capability._detect_yes_no_fast("sind lichter an?") is True
    assert yes_no_capability._detect_yes_no_fast("gibt es fenster?") is True
    assert yes_no_capability._detect_yes_no_fast("welche lichter sind an?") is False
    assert yes_no_capability._detect_yes_no_fast("schalte licht an") is False
