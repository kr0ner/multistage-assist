"""Integration tests for time normalization in Semantic Cache."""

import pytest
from unittest.mock import MagicMock, patch
from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability
from multistage_assist.utils.german_utils import normalize_for_cache


pytestmark = pytest.mark.integration


def test_normalize_time_expressions():
    """Test that various time expressions are normalized to '10 Minuten'."""
    test_cases = [
        ("Schalte das Licht in 5 Minuten aus", "Schalte das Licht in 10 Minuten aus"),
        ("Mach das Licht in 37 Sekunden an", "Mach das Licht in 10 Minuten an"),
        ("Erinnere mich in 2 Stunden", "Erinnere mich in 10 Minuten"),
        ("Timer für 15 Minuten", "Timer für 10 Minuten"),
        ("Licht an für 30 Sekunden", "Licht an für 10 Minuten"),
        ("Rollo runter um 15 Uhr", "Rollo runter um 10 Uhr"), # "um 15 Uhr" -> "um 10 Uhr" (cache norm)
    ]

    for input_text, expected in test_cases:
        normalized, _ = normalize_for_cache(input_text)
        assert normalized == expected, f"Failed to normalize '{input_text}'"


@pytest.mark.asyncio
async def test_cache_lookup_uses_normalized_text(hass):
    """Test that cache lookup uses normalized text."""
    # Mock config
    config = {
        "cache_enabled": True,
    }
    
    capability = SemanticCacheCapability(hass, config)
    
    # Mock the _addon_url and aiohttp session (partially) or just mock lookup internal calls
    # Ideally we'd like to see what text is sent to the API.
    # Since lookup calls self._addon_url, we can't easily spy on the HTTP request without patching aiohttp.
    
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json.return_value = {"found": False} # We minimal return
        mock_post.return_value.__aenter__.return_value = mock_resp
        
        await capability.lookup("Schalte Licht in 37 Sekunden aus")
        
        # Verify the payload sent to API contained the NORMALIZED text
        call_args = mock_post.call_args
        kw = call_args.kwargs
        payload = kw["json"]
        
        assert payload["query"] == "Schalte Licht in 10 Minuten aus", \
            "Cache lookup did not use normalized text!"
