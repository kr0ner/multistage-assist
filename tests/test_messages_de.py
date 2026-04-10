"""Tests for German messaging constants."""

import pytest
from unittest.mock import patch
from multistage_assist.constants.messages_de import get_domain_confirmation

def test_singular_confirmation():
    """Test that singular confirmation uses singular verb forms."""
    msg = get_domain_confirmation(
        domain="light",
        action="on",
        name="Das Licht",
        state="on",
        is_plural=False
    )
    assert "sind " not in msg
    assert "laufen " not in msg
    assert "wurden " not in msg

def test_plural_confirmation():
    """Test that plural confirmation correctly replaces verbs."""
    # Run multiple times to catch random template variance
    for _ in range(10):
        msg = get_domain_confirmation(
            domain="light",
            action="on",
            name="Die Lichter",
            state="on",
            is_plural=True
        )
        
        # Singular verbs should NOT appear in plural form
        assert " ist " not in msg
        assert " wurde " not in msg
