"""Test timer description extraction feature."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from multistage_assist.capabilities.timer import TimerCapability
from homeassistant.components import conversation


@pytest.mark.asyncio
async def test_timer_description_extraction():
    """Test that timer descriptions are extracted correctly from natural language."""
    hass = MagicMock()
    config = {
        "stage1_ip": "127.0.0.1",
        "stage1_port": 11434,
        "stage1_model": "qwen3:4b-instruct",
    }

    timer_cap = TimerCapability(hass, config)

    test_cases = [
        (
            "Setze einen Timer für 5 Minuten der mich daran erinnert, dass die Nudeln fertig sind",
            "Nudeln",
        ),
        ("Timer für 10 Minuten damit die Pizza nicht verbrennt", "Pizza"),
        ("5 Minuten Timer für den Tee", "Tee"),
        ("Timer für 3 Minuten", ""),  # No description
        ("Stelle einen Timer auf 20 Minuten", ""),  # No description
    ]

    for input_text, expected_desc in test_cases:
        desc = await timer_cap._extract_description(input_text)
        print(f"Input: {input_text}")
        print(f"Expected: '{expected_desc}', Got: '{desc}'")

        # Fuzzy match since LLM might return slightly different phrasing
        if expected_desc:
            assert (
                expected_desc.lower() in desc.lower()
                or desc.lower() in expected_desc.lower()
            ), f"Expected '{expected_desc}' in description, got '{desc}'"
        # For empty expectation, just ensure it's short or empty
        else:
            assert len(desc) < 5, f"Expected empty/very short description, got '{desc}'"


@pytest.mark.asyncio
async def test_timer_with_description_end_to_end(hass):
    """Test timer with description is passed to Android intent."""
    config = {
        "stage1_ip": "127.0.0.1",
        "stage1_port": 11434,
        "stage1_model": "qwen3:4b-instruct",
    }

    # Mock mobile services
    hass.services.async_services = MagicMock(
        return_value={
            "notify": {
                "mobile_app_sm_a566b": {"description": "SM-A566B"},
            }
        }
    )

    # Mock service call
    hass.services.async_call = AsyncMock()

    timer_cap = TimerCapability(hass, config)

    user_input = MagicMock()
    user_input.text = "Setze einen Timer für 5 Minuten für die Nudeln"
    user_input.language = "de"

    # Run the capability
    result = await timer_cap.run(
        user_input=user_input,
        intent_name="HassTimerSet",
        slots={"duration": "5 Minuten", "name": "mobile_app_sm_a566b"},
    )

    # Verify service was called
    assert hass.services.async_call.called

    # Get the call args
    call_args = hass.services.async_call.call_args
    payload = call_args[0][2] if len(call_args[0]) > 2 else call_args[1]

    # Verify description is in the intent extras
    intent_extras = payload["data"]["intent_extras"]
    print(f"Intent extras: {intent_extras}")

    # Should contain MESSAGE extra with description
    assert (
        "android.intent.extra.alarm.MESSAGE:" in intent_extras
    ), f"Expected description in intent extras: {intent_extras}"

