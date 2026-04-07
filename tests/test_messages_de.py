"""Tests for German messaging constants."""

import pytest
from multistage_assist.constants.messages_de import get_domain_confirmation

def test_singular_confirmation():
    """Test that singular confirmation uses 'ist', 'läuft' etc."""
    msg = get_domain_confirmation(
        domain="light",
        action="on",
        name="Das Licht",
        state="on",
        is_plural=False
    )
    # The template might say 'ist an' or 'wurde eingeschaltet'
    # We assert it definitely doesn't say 'sind an'
    assert "sind " not in msg
    assert "laufen " not in msg
    assert "wurden " not in msg

def test_plural_confirmation():
    """Test that plural confirmation correctly replaces verbs."""
    msg = get_domain_confirmation(
        domain="light",
        action="on",
        name="Die Lichter",
        state="on",
        is_plural=True
    )
    
    # Depending on the random template chosen, it should contain a pluralized word
    # Let's override the random choice implicitly by testing the text
    # Pluralization replaces "ist" with "sind", "wurde" with "wurden"
    assert " ist " not in msg
    assert " wurde " not in msg
    assert " kümmert sich " not in msg
    assert " geht " not in msg
    
    # Verify that at least some plural verb is present (sind, wurden, leuchten, etc)
    plural_verbs = ["sind", "wurden", "leuchten", "laufen", "machen", "kümmern sich", "schließen", "öffnen", "gehen", "stehen"]
    words = msg.lower().split()
    assert any(verb in words for verb in plural_verbs) or "erledigt." in msg  # erledigt has no verb
