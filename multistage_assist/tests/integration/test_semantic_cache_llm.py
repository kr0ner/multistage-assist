"""Integration tests for SemanticCacheCapability with semantic anchors.

These tests verify the semantic anchor system:
- Anchors are created for each intent × area
- New commands hit anchors and escalate to LLM
- After LLM processes, real entries beat anchors

Requires:
- Ollama running with bge-m3 model
- sentence-transformers installed with BAAI/bge-reranker-v2-m3

Run with: pytest tests/integration/test_semantic_cache_llm.py -v -m integration
"""

import os
import pytest
import numpy as np

from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability
from . import get_llm_config, OLLAMA_HOST, OLLAMA_PORT

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def llm_cache(hass):
    """Create semantic cache with real Ollama embeddings and reranker."""
    config = get_llm_config()
    config["cache_enabled"] = True
    config["embedding_model"] = "bge-m3"
    config["reranker_model"] = "BAAI/bge-reranker-v2-m3"
    config["reranker_enabled"] = True
    config["reranker_threshold"] = 0.73
    config["reranker_device"] = "cpu"  # Force CPU to avoid GPU OOM
    config["vector_search_threshold"] = 0.4
    config["vector_search_top_k"] = 10  # More candidates for anchor matching

    # Use embedding config from Ollama settings
    config["embedding_ip"] = OLLAMA_HOST
    config["embedding_port"] = OLLAMA_PORT

    cache = SemanticCacheCapability(hass, config)

    # Skip anchor initialization for faster tests (we test it explicitly)
    cache._anchors_initialized = True
    cache._loaded = True

    return cache


@pytest.fixture
def llm_cache_with_anchors(hass):
    """Create semantic cache with anchors initialized."""
    config = get_llm_config()
    config["cache_enabled"] = True
    config["embedding_model"] = "bge-m3"
    config["reranker_model"] = "BAAI/bge-reranker-v2-m3"
    config["reranker_enabled"] = True
    config["reranker_threshold"] = 0.73
    config["reranker_device"] = "cpu"
    config["vector_search_threshold"] = 0.4
    config["vector_search_top_k"] = 10
    config["embedding_ip"] = OLLAMA_HOST
    config["embedding_port"] = OLLAMA_PORT

    cache = SemanticCacheCapability(hass, config)
    # Anchors will be initialized on first lookup
    return cache


# ============================================================================
# REAL ENTRY TESTS (No anchors - direct caching)
# ============================================================================


@pytest.mark.asyncio
async def test_exact_match_returns_cached(llm_cache, hass):
    """Test that exact or near-exact matches return cached result."""
    await llm_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    result = await llm_cache.lookup("Schalte das Licht in der Küche an")

    assert result is not None
    assert result["intent"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_synonym_returns_cached(llm_cache, hass):
    """Test that synonymous phrasing returns cached result."""
    await llm_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    synonyms = [
        "Mach das Licht in der Küche an",
        "Lampe in der Küche einschalten",
    ]

    for query in synonyms:
        result = await llm_cache.lookup(query)
        assert result is not None, f"Expected match for synonym: '{query}'"
        assert result["intent"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_opposite_action_blocked(llm_cache, hass):
    """Test that opposite actions are blocked."""
    await llm_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    result = await llm_cache.lookup("Schalte das Licht in der Küche aus")

    # Should be blocked by reranker (score too low)
    assert result is None


@pytest.mark.asyncio
async def test_different_room_blocked(llm_cache, hass):
    """Test that different rooms are blocked."""
    await llm_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    result = await llm_cache.lookup("Schalte das Licht im Wohnzimmer an")

    # Should be blocked - different room
    assert result is None


# ============================================================================
# ANCHOR BEHAVIOR TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_anchors_are_created(llm_cache_with_anchors, hass):
    """Test that anchors are generated on initialization."""
    # Trigger anchor initialization via lookup
    await llm_cache_with_anchors.lookup("test query")

    stats = await llm_cache_with_anchors.get_stats()

    # Should have created anchors
    assert stats["anchor_count"] > 0
    assert stats["real_entries"] == 0  # No real entries yet


@pytest.mark.asyncio
async def test_new_command_hits_anchor_escalates(llm_cache_with_anchors, hass):
    """Test that new commands hit anchors and escalate to LLM."""
    # First lookup triggers anchor creation
    # Then this query should match an anchor → escalate (return None)
    result = await llm_cache_with_anchors.lookup("Mach das Licht in der Küche heller")

    # Should return None (anchor hit = escalate to LLM)
    assert result is None

    stats = await llm_cache_with_anchors.get_stats()
    assert stats["anchor_escalations"] > 0


@pytest.mark.asyncio
async def test_real_entry_beats_anchor(llm_cache_with_anchors, hass):
    """Test that real cached entries take priority over anchors."""
    # Store a real command (after anchors are initialized)
    await llm_cache_with_anchors.store(
        text="Mach die Lampe in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # Same command should now return the real entry, not anchor
    result = await llm_cache_with_anchors.lookup("Mach die Lampe in der Küche an")

    # Real entry should beat anchor because exact match scores higher
    assert result is not None
    assert result["entity_ids"] == ["light.kuche"]


@pytest.mark.asyncio
async def test_brightness_hits_brightness_anchor(llm_cache_with_anchors, hass):
    """Test that brightness commands hit brightness anchors, not on/off."""
    # Store a TurnOn command
    await llm_cache_with_anchors.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # Brightness query should hit HassLightSet anchor, not TurnOn entry
    result = await llm_cache_with_anchors.lookup("Mach das Licht in der Küche heller")

    # Should escalate (anchor hit) rather than wrongly return TurnOn
    assert result is None

    stats = await llm_cache_with_anchors.get_stats()
    assert stats["anchor_escalations"] >= 1


@pytest.mark.asyncio
async def test_off_hits_off_anchor_not_on(llm_cache_with_anchors, hass):
    """Test that 'off' commands hit TurnOff anchors, not cached 'on' commands."""
    # Store an 'on' command
    await llm_cache_with_anchors.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # 'Off' query should hit HassTurnOff anchor
    result = await llm_cache_with_anchors.lookup("Schalte das Licht in der Küche aus")

    # Should escalate to LLM rather than wrongly return 'on' action
    assert result is None


# ============================================================================
# STATS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_stats_include_anchor_info(llm_cache_with_anchors, hass):
    """Test that stats include anchor count."""
    await llm_cache_with_anchors.lookup("trigger anchor init")

    stats = await llm_cache_with_anchors.get_stats()

    assert "anchor_count" in stats
    assert "real_entries" in stats
    assert "anchor_escalations" in stats
    assert stats["anchor_count"] > 0
