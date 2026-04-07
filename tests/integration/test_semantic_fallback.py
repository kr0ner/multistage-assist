"""Integration tests for semantic fallback in KeywordIntentCapability."""

import os
import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability


pytestmark = pytest.mark.integration

# Cache Addon configuration
CACHE_HOST = os.getenv("CACHE_HOST", "192.168.178.2")
CACHE_PORT = int(os.getenv("CACHE_PORT", "9876"))


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability with cache addon."""
    config = {
        "cache_addon_ip": CACHE_HOST,
        "cache_addon_port": CACHE_PORT,
    }
    return KeywordIntentCapability(hass, config)


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


class TestSemanticFallback:
    """Test semantic domain detection when keywords fail."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text,expected_domain", [
        # Queries WITHOUT typical German keywords (Licht/Lampe/Rollo)
        # Using abstract descriptions that require semantic matching
        
        # Light
        ("Mache es heller hier", "light"),
        ("Es ist so finster", "light"),
        
        # Cover
        ("Zieh die Vorhänge zu", "cover"),
        ("Sichtschutz schließen", "cover"),
        
        # Climate
        ("Mir ist eiskalt", "climate"),
        ("Ich friere", "climate"),
        
        # Vacuum
        ("Boden ist dreckig", "vacuum"),
        ("Krümel entfernen", "vacuum"),
        
        # Media
        ("Beschalle das Zimmer", "media_player"),
        ("Mach mal Krach", "media_player"),
        
        # Fan
        ("Mache Wind", "fan"),
    ])
    async def test_semantic_domain_detection(
        self, keyword_intent_capability, text, expected_domain
    ):
        """Test fallback to semantic matching."""
        # Ensure _detect_domain (fuzzy/exact) returns None for these
        # This confirms we are actually testing the fallback.
        # Note: If this fails, the test case is too easy and matched via keywords.
        assert keyword_intent_capability._detect_domain(text) is None, \
            f"Text '{text}' surprisingly matched via keywords! Use a harder test case."

        # Run _semantic_match directly to verify cache addon logic
        domain = await keyword_intent_capability._semantic_match(text)
        
        assert domain == expected_domain, \
            f"Expected domain '{expected_domain}' for '{text}', got '{domain}'"
