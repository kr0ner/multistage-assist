"""Integration tests for ClarificationCapability with real LLM."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.clarification import ClarificationCapability


pytestmark = pytest.mark.integration  # Mark all tests as integration tests


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def clarification_capability(hass):
    """Create clarification capability instance with real LLM."""
    config = {
        "stage1_ip": "127.0.0.1",  # Real LLM server
        "stage1_port": 11434,
        "stage1_model": "qwen3:4b-instruct",
    }
    return ClarificationCapability(hass, config)


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
    "user_text,expected_count,description",
    [
        # Single commands (should return 1)
        ("Schalte das Licht an", 1, "Simple light on"),
        ("Mache das Rollo runter", 1, "Simple cover down"),
        ("Stelle einen Timer auf 5 Minuten", 1, "Simple timer"),
        ("Wie warm ist es?", 1, "Simple question"),
        # Multi-commands with "und" (should return 2+)
        ("Licht an und Rollo zu", 2, "Two commands with und"),
        (
            "Schalte das Licht im Bad an und im Flur aus",
            2,
            "Light on/off different rooms",
        ),
        ("Mache Licht an, Rollo runter und Heizung auf 22", 3, "Three commands"),
        # Implicit actions (should be expanded)
        ("Im Büro ist es zu dunkel", 1, "Implicit: make brighter"),
        ("Zu hell hier", 1, "Implicit: make darker"),
        # Duration preservation
        ("Schalte das Licht für 10 Minuten an", 1, "Duration preserved"),
        (
            "Licht für 5 Minuten an und Timer auf 2 Minuten",
            2,
            "Two commands with durations",
        ),
        # Complex multi-room
        (
            "Licht in der Küche an und im Wohnzimmer aus",
            2,
            "Multi-room opposite actions",
        ),
        (
            "Heizung im Schlafzimmer auf 18 und im Büro auf 21",
            2,
            "Multi-room different values",
        ),
        # Edge cases
        ("Alle Lichter aus", 1, "Global command"),
        ("Licht", 1, "Minimal command"),
    ],
)
async def test_clarification_splitting(
    clarification_capability, user_text, expected_count, description
):
    """Test clarification with real LLM."""
    user_input = make_input(user_text)
    result = await clarification_capability.run(user_input)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert (
        len(result) == expected_count
    ), f"{description}: Expected {expected_count} commands, got {len(result)}: {result}"

    # Each result should be a non-empty string
    for cmd in result:
        assert isinstance(cmd, str), f"Expected string command, got {type(cmd)}"
        assert cmd.strip(), "Command should not be empty"


@pytest.mark.parametrize(
    "user_text,should_preserve",
    [
        ("Schalte das Licht für 10 Minuten an", "10 Minuten"),
        ("Timer auf 5 Minuten", "5 Minuten"),
        ("Licht für eine Stunde an", "eine Stunde"),
        ("Für 30 Sekunden anschalten", "30 Sekunden"),
    ],
)
async def test_duration_preservation(
    clarification_capability, user_text, should_preserve
):
    """Test that durations are preserved in clarified output."""
    user_input = make_input(user_text)
    result = await clarification_capability.run(user_input)

    assert isinstance(result, list)
    assert len(result) > 0

    # Check if duration is preserved in output
    combined = " ".join(result).lower()
    assert (
        should_preserve.lower() in combined
    ), f"Duration '{should_preserve}' not preserved in output: {result}"


async def test_opposite_actions_split(clarification_capability):
    """Test that opposite actions in different locations are split."""
    test_cases = [
        "Licht im Bad an und im Flur aus",
        "Rollo im Wohnzimmer auf und im Schlafzimmer zu",
        "Heizung im Büro an und in der Küche aus",
    ]

    for text in test_cases:
        user_input = make_input(text)
        result = await clarification_capability.run(user_input)

        assert (
            len(result) == 2
        ), f"Expected 2 commands for '{text}', got {len(result)}: {result}"


async def test_implicit_to_explicit(clarification_capability):
    """Test that implicit commands are made explicit."""
    test_cases = [
        ("Es ist zu dunkel", "heller"),
        ("Zu hell hier", "dunkler"),
        ("Im Büro ist es zu dunkel", "heller"),
    ]

    for user_text, expected_word in test_cases:
        user_input = make_input(user_text)
        result = await clarification_capability.run(user_input)

        assert len(result) > 0
        combined = " ".join(result).lower()
        assert (
            expected_word in combined
        ), f"Expected '{expected_word}' in clarification of '{user_text}', got: {result}"
