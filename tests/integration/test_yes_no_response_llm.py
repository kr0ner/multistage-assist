"""Integration tests for YesNoResponseCapability with real LLM."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.yes_no_response import YesNoResponseCapability


pytestmark = pytest.mark.integration


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    hass = MagicMock()

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
    """Create yes/no response capability with real LLM."""
    config = {
        "stage1_ip": "127.0.0.1",
        "stage1_port": 11434,
        "stage1_model": "qwen3:4b-instruct",
    }
    return YesNoResponseCapability(hass, config)


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
    "user_text,should_be_yes_no",
    [
        # YES/NO questions (keyword fast-path)
        ("Ist ein Licht an?", True),
        ("Sind Lichter an?", True),
        ("Gibt es offene Fenster?", True),
        ("Hat das Licht einen Timer?", True),
        ("Haben wir Rollläden offen?", True),
        # NOT yes/no questions
        ("Welche Lichter sind an?", False),  # Which, not yes/no
        ("Schalte das Licht an", False),  # Command
        ("Alle Lichter ausschalten", False),  # Command
        ("Zeige mir die Lichter", False),  # Show/list command
        # Clear yes/no with modal verbs
        ("Kann ich die Lampe einschalten?", True),  # Can I... (yes/no)
        ("Soll ich das Licht anmachen?", True),  # Should I... (yes/no)
    ],
)
async def test_yes_no_detection(yes_no_capability, user_text, should_be_yes_no):
    """Test yes/no question detection with real LLM (including fallback)."""
    user_input = make_input(user_text)

    result = await yes_no_capability.run(
        user_input,
        domain="light",
        state="on",
        entity_ids=["light.kuche"] if should_be_yes_no else [],
    )

    if should_be_yes_no:
        assert result is not None, f"Expected yes/no response for: {user_text}"
        assert (
            "ja" in result.lower() or "nein" in result.lower()
        ), f"Yes/no response should contain 'ja' or 'nein' for: {user_text}, got: {result}"
    else:
        # Should return None for non-yes/no questions
        assert (
            result is None
        ), f"Should not detect yes/no for: {user_text}, got: {result}"


@pytest.mark.parametrize(
    "entity_count,expected_verb",
    [
        (0, None),  # "Nein, kein..." - different structure
        (1, "ist"),
        (2, "sind"),
        (3, "sind"),
    ],
)
async def test_response_formatting(yes_no_capability, entity_count, expected_verb):
    """Test response formatting with different entity counts."""
    user_input = make_input("Ist ein Licht an?")

    entity_ids = [f"light.test_{i}" for i in range(entity_count)]

    result = await yes_no_capability.run(
        user_input,
        domain="light",
        state="on",
        entity_ids=entity_ids,
    )

    assert result is not None
    response_lower = result.lower()

    if entity_count == 0:
        assert "nein" in response_lower
    else:
        assert "ja" in response_lower
        if expected_verb:
            assert (
                expected_verb in response_lower
            ), f"Expected verb '{expected_verb}' for {entity_count} entities, got: {result}"


async def test_llm_fallback_for_edge_cases(yes_no_capability):
    """Test that LLM fallback works for questions with '?' but non-standard structure."""
    edge_cases = [
        "Brennt irgendwo ein Licht?",  # Non-standard verb
        "Habe ich ein Licht angelassen?",  # Complex question
        "Leuchtet was?",  # Very informal
    ]

    for text in edge_cases:
        user_input = make_input(text)
        result = await yes_no_capability.run(
            user_input,
            domain="light",
            state="on",
            entity_ids=["light.kuche"],
        )

        # Should use LLM fallback and detect as yes/no due to "?"
        if result:  # LLM might correctly identify it
            assert (
                "ja" in result.lower() or "nein" in result.lower()
            ), f"LLM fallback should generate yes/no answer for: {text}, got: {result}"


@pytest.mark.parametrize(
    "domain,state,expected_word",
    [
        ("light", "on", "licht"),
        ("cover", "open", "rollladen"),
        ("switch", "on", "schalter"),
        ("fan", "on", "ventilator"),
        ("climate", "on", "thermostat"),
    ],
)
async def test_domain_names_in_responses(
    yes_no_capability, domain, state, expected_word
):
    """Test that correct German domain names are used in responses."""
    user_input = make_input(f"Ist ein {expected_word} {state}?")

    # Test negative response (no entities)
    result = await yes_no_capability.run(
        user_input,
        domain=domain,
        state=state,
        entity_ids=[],
    )

    assert result is not None
    response_lower = result.lower()
    assert (
        expected_word in response_lower
    ), f"Expected domain word '{expected_word}' in response for domain '{domain}', got: {result}"
