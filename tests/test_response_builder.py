"""Tests for response_builder utilities.

Tests German response generation including confirmations, 
entity list formatting, and TTS normalization.
"""

import pytest

from multistage_assist.utils.response_builder import (
    join_names,
    format_entity_list,
    get_domain_name,
    build_confirmation,
    build_question,
    build_selection_question,
    build_error,
    normalize_for_tts,
)


# ============================================================================
# JOIN NAMES TESTS
# ============================================================================

def test_join_names_empty():
    """Test join_names with empty list."""
    assert join_names([]) == ""


def test_join_names_single():
    """Test join_names with single item."""
    assert join_names(["Küche"]) == "Küche"


def test_join_names_two():
    """Test join_names with two items."""
    assert join_names(["Küche", "Bad"]) == "Küche und Bad"


def test_join_names_multiple():
    """Test join_names with multiple items."""
    assert join_names(["A", "B", "C"]) == "A, B und C"
    assert join_names(["A", "B", "C", "D"]) == "A, B, C und D"


def test_join_names_custom_conjunction():
    """Test join_names with custom conjunction."""
    assert join_names(["A", "B"], conjunction="oder") == "A oder B"


# ============================================================================
# FORMAT ENTITY LIST TESTS
# ============================================================================

def test_format_entity_list_short():
    """Test format_entity_list with short list."""
    assert format_entity_list(["A", "B"]) == "A und B"


def test_format_entity_list_truncated():
    """Test format_entity_list truncates long lists."""
    result = format_entity_list(["A", "B", "C", "D", "E", "F"])
    assert "weiteres" in result or "weitere" in result
    assert "A" in result


# ============================================================================
# DOMAIN NAME TESTS
# ============================================================================

def test_get_domain_name_known():
    """Test get_domain_name returns German names."""
    assert get_domain_name("light") == "Licht"
    assert get_domain_name("cover") == "Rollo"
    assert get_domain_name("switch") == "Schalter"


def test_get_domain_name_unknown():
    """Test get_domain_name with unknown domain."""
    result = get_domain_name("unknown_domain")
    # Should return something (capitalized or original)
    assert result is not None


# ============================================================================
# BUILD CONFIRMATION TESTS - High Priority (source of bugs)
# ============================================================================

def test_build_confirmation_turn_on():
    """Test build_confirmation for HassTurnOn."""
    result = build_confirmation(
        intent_name="HassTurnOn",
        device_names=["Küche"],
        domain="light",
    )
    assert "an" in result.lower()
    assert "Küche" in result


def test_build_confirmation_turn_off():
    """Test build_confirmation for HassTurnOff."""
    result = build_confirmation(
        intent_name="HassTurnOff",
        device_names=["Küche"],
        domain="light",
    )
    assert "aus" in result.lower()
    assert "Küche" in result


def test_build_confirmation_multiple_devices():
    """Test build_confirmation with multiple devices."""
    result = build_confirmation(
        intent_name="HassTurnOn",
        device_names=["Küche", "Bad", "Büro"],
        domain="light",
    )
    assert "Küche" in result
    assert "und" in result


def test_build_confirmation_with_brightness():
    """Test build_confirmation with brightness parameter."""
    result = build_confirmation(
        intent_name="HassLightSet",
        device_names=["Büro"],
        domain="light",
        params={"brightness": 50},
    )
    assert "Büro" in result
    # Should mention brightness percentage
    assert "50" in result or "%" in result


def test_build_confirmation_with_duration():
    """Test build_confirmation with duration parameter."""
    result = build_confirmation(
        intent_name="HassTurnOn",
        device_names=["Licht"],
        domain="light",
        params={"duration_seconds": 600},  # 10 minutes
    )
    assert "Licht" in result
    # May contain duration info
    assert result  # At minimum, should return something


def test_build_confirmation_cover_position():
    """Test build_confirmation for cover position."""
    result = build_confirmation(
        intent_name="HassSetPosition",
        device_names=["Rollladen"],
        domain="cover",
        params={"position": 50},
    )
    assert "Rollladen" in result


def test_build_confirmation_empty_devices():
    """Test build_confirmation with empty device list."""
    result = build_confirmation(
        intent_name="HassTurnOn",
        device_names=[],
        domain="light",
    )
    # Should handle gracefully
    assert result is not None


# ============================================================================
# BUILD QUESTION TESTS
# ============================================================================

def test_build_question_simple():
    """Test build_question for area."""
    result = build_question("area")
    assert "?" in result


def test_build_question_with_context():
    """Test build_question with context."""
    result = build_question("device", context="im Wohnzimmer")
    assert "?" in result


# ============================================================================
# BUILD SELECTION QUESTION TESTS
# ============================================================================

def test_build_selection_question():
    """Test build_selection_question with options."""
    result = build_selection_question(
        field="device",
        options=["Handy", "Tablet"],
    )
    assert "?" in result
    assert "Handy" in result or "Tablet" in result


# ============================================================================
# BUILD ERROR TESTS
# ============================================================================

def test_build_error_known():
    """Test build_error with known error type."""
    result = build_error("not_found")
    assert "nicht gefunden" in result.lower() or "gefunden" in result.lower()


def test_build_error_with_details():
    """Test build_error with details."""
    result = build_error("not_found", details="Küche")
    assert result  # Should include details somehow


def test_build_error_unknown():
    """Test build_error with unknown error type."""
    result = build_error("unknown_error_type")
    assert result is not None  # Should fallback gracefully


# ============================================================================
# TTS NORMALIZATION TESTS
# ============================================================================

def test_normalize_for_tts_temperature():
    """Test normalize_for_tts for temperature."""
    result = normalize_for_tts("22°C")
    assert "Grad" in result
    assert "22" in result


def test_normalize_for_tts_percentage():
    """Test normalize_for_tts for percentage."""
    result = normalize_for_tts("50%")
    assert "Prozent" in result
    assert "50" in result


def test_normalize_for_tts_power():
    """Test normalize_for_tts for power units."""
    result = normalize_for_tts("100W")
    assert "Watt" in result
    assert "100" in result


def test_normalize_for_tts_plain_text():
    """Test normalize_for_tts with plain text."""
    result = normalize_for_tts("Licht ist an")
    assert result == "Licht ist an"


def test_normalize_for_tts_multiple_units():
    """Test normalize_for_tts with multiple units."""
    result = normalize_for_tts("Verbrauch: 5kWh bei 22°C")
    assert "Kilowattstunden" in result
    assert "Grad" in result
