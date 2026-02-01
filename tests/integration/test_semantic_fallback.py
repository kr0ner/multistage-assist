"""Integration tests for semantic fallback in KeywordIntentCapability."""

import os
import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation

from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability


pytestmark = pytest.mark.integration

# Reranker configuration
RERANKER_HOST = os.getenv("RERANKER_HOST", "192.168.178.2")
RERANKER_PORT = int(os.getenv("RERANKER_PORT", "9876"))


@pytest.fixture
def hass():
    """Mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def keyword_intent_capability(hass):
    """Create keyword intent capability with reranker enabled."""
    config = {
        "reranker_enabled": True,
        "reranker_ip": RERANKER_HOST,
        "reranker_port": RERANKER_PORT,
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
        # Using English or abstract descriptions that reranker understands
        
        # Light
        ("Make it bright here", "light"),
        ("Illuminate the room", "light"),
        ("Es ist zu dunkel hier", "light"),  # implicit
        
        # Cover
        ("Open the window shades", "cover"),
        ("I need some privacy", "cover"),  # might map to closing blinds?
        
        # Climate
        ("It is freezing in here", "climate"),
        ("I need heat", "climate"),  # "warm" is a sensor keyword!
        ("I am cold", "climate"),
        
        # Vacuum
        ("Clean the floor", "vacuum"),
        ("Start cleaning", "vacuum"),
        
        # Media
        ("Let us hear something", "media_player"),  # "audio"/"tracks" matched keywords
        ("Stop the noise", "media_player"),
    ])
    async def test_semantic_domain_detection(
        self, keyword_intent_capability, text, expected_domain
    ):
        """Test fallback to semantic matching."""
        # Ensure _detect_domain (fuzzy/exact) returns None for these
        # This confirms we are actually testing the fallback
        assert keyword_intent_capability._detect_domain(text) is None, \
            f"Text '{text}' surprisingly matched via keywords! Use a harder test case."

        # Run full pipeline which includes fallback
        user_input = make_input(text)
        
        # We only care that it finds the domain and starts LLM prompt
        # The LLM prompt might fail since we mock nothing, but run() calls _detect_domain then _semantic_match
        
        # To test _semantic_match specifically without invoking LLM (which needs separate mocking)
        # we can call _semantic_match directly or just check if run() gets past domain detection.
        
        domain = await keyword_intent_capability._semantic_match(text)
        
        assert domain == expected_domain, \
            f"Expected domain '{expected_domain}' for '{text}', got '{domain}'"

