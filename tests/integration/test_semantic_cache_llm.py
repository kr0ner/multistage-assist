"""These tests verify anchor escalation, learning prioritization, and
cross-domain precision. Uses a distilled German model via the cache-addon.
"""

import os
import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability
from multistage_assist.utils.semantic_cache_builder import CACHE_VERSION
from .test_fixtures import (
    generate_test_anchors,
    TEST_AREAS,
    ANCHOR_PATTERNS
)

pytestmark = pytest.mark.integration

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
CACHE_HOST = os.getenv("CACHE_HOST", "127.0.0.1")
CACHE_PORT = int(os.getenv("CACHE_PORT", 9876))


def get_llm_config():
    """Get Ollama config from environment."""
    return {
        "cache_addon_ip": CACHE_HOST,
        "cache_addon_port": CACHE_PORT,
        "embedding_ip": OLLAMA_HOST,
        "embedding_port": OLLAMA_PORT,
        "embedding_model": os.getenv("OLLAMA_MODEL", "never use "),
    }


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def clean_storage(tmp_path):
    """Create a clean .storage directory for each test."""
    storage_dir = tmp_path / ".storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def llm_cache(clean_storage, hass):
    """Create semantic cache with real Ollama embeddings and cache addon."""
    config = get_llm_config()
    
    # Mock hass config path to point to the clean temp storage
    def mock_path(*args):
        if args and args[0] == ".storage":
            return str(clean_storage)
        return str(clean_storage.parent)
    hass.config.path = MagicMock(side_effect=mock_path)

    config["cache_enabled"] = True
    config["cache_threshold"] = 0.82
    config["vector_search_threshold"] = 0.4
    config["vector_search_top_k"] = 10

    cache = SemanticCacheCapability(hass, config)

    # Initialize properly (loads empty cache)
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cache.async_startup())
    except RuntimeError:
        asyncio.run(cache.async_startup())

    return cache


@pytest.fixture
async def llm_cache_with_anchors(clean_storage, hass):
    """Create semantic cache with anchors initialized."""
    config = get_llm_config()
    
    # Mock hass config path to point to the clean temp storage
    def mock_path(*args):
        if args and args[0] == ".storage":
            return str(clean_storage)
        return str(clean_storage.parent)
    hass.config.path = MagicMock(side_effect=mock_path)

    # Pre-generate anchors into the temp storage
    cache_cap = SemanticCacheCapability(hass, config)
    anchors_data = await generate_test_anchors(embed_func=cache_cap._get_embedding)
    
    anchor_file = clean_storage / "multistage_assist_anchors.json"
    with open(anchor_file, "w") as f:
        json.dump(anchors_data, f)

    config["cache_enabled"] = True
    config["cache_threshold"] = 0.82
    config["vector_search_threshold"] = 0.4
    config["vector_search_top_k"] = 10

    cache = SemanticCacheCapability(hass, config)

    # Initialize properly (loads anchors from the file we just created)
    await cache.async_startup()

    return cache


# ============================================================================
# REAL ENTRY TESTS (No anchors - direct caching)
# ============================================================================


@pytest.mark.asyncio
async def test_exact_match_returns_cached(llm_cache, hass):
    """Test that exact or near-exact matches return cached result."""
    # Use a room NOT in TEST_AREAS to avoid background anchor collisions
    await llm_cache.store(
        text="Schalte das Licht im Hobbyraum an",
        intent="HassTurnOn",
        entity_ids=["light.hobbyraum"],
        slots={"area": "Hobbyraum"},
        verified=True,
    )

    # Exact match
    result = await llm_cache.lookup("Schalte das Licht im Hobbyraum an")
    assert result is not None
    assert result["intent"] == "HassTurnOn"
    assert result["entity_ids"] == ["light.hobbyraum"]
    assert result["source"] == "learned"

    # Near-exact match (different case and trailing space)
    result = await llm_cache.lookup("schalte DAS licht im HOBBYRAUM an ")
    assert result is not None
    assert result["intent"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_fuzzy_match_returns_cached(llm_cache, hass):
    """Test that fuzzy matches return cached result."""
    # Use a unique string to avoid any collisions
    await llm_cache.store(
        text="Mache das Licht im Dachstudio an",
        intent="HassTurnOn",
        entity_ids=["light.dachstudio"],
        slots={"area": "Dachstudio"},
        verified=True,
    )

    # Fuzzy match (minor variation)
    result = await llm_cache.lookup("Mach mal das Licht im Dachstudio an")
    assert result is not None
    assert result["intent"] == "HassTurnOn"


@pytest.mark.asyncio
async def test_opposite_action_blocked(llm_cache, hass):
    """Test that opposite actions are blocked."""
    await llm_cache.store(
        text="Schalte das Licht im Hobbyraum an",
        intent="HassTurnOn",
        entity_ids=["light.hobbyraum"],
        slots={"area": "Hobbyraum"},
        verified=True,
    )

    result = await llm_cache.lookup("Schalte das Licht im Hobbyraum aus")

    # Should be blocked by high precision threshold
    assert result is None


@pytest.mark.asyncio
async def test_different_room_blocked(llm_cache, hass):
    """Test that different rooms are blocked."""
    await llm_cache.store(
        text="Schalte das Licht im Hobbyraum an",
        intent="HassTurnOn",
        entity_ids=["light.hobbyraum"],
        slots={"area": "Hobbyraum"},
        verified=True,
    )

    result = await llm_cache.lookup("Schalte das Licht im Gästebad an")

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

    stats = llm_cache_with_anchors.get_stats()

    # Should have created anchors
    assert stats["anchor_count"] > 0
    assert stats.get("real_entries", 0) == 0  # No real entries yet


@pytest.mark.asyncio
async def test_new_command_hits_anchor_escalates(llm_cache_with_anchors, hass):
    """Test that new commands hit anchors and escalate to LLM."""
    # This query matches an anchor in kitchen -> escalate (return None)
    # Using a slightly different room name to avoid remote pollution if possible
    # but still hitting the pre-generated synthetic anchors
    result = await llm_cache_with_anchors.lookup("Mach das Licht im Büro heller")

    # Should return None (anchor hit = escalate to LLM)
    assert result is None

    stats = llm_cache_with_anchors.get_stats()
    assert stats["anchor_escalations"] > 0


@pytest.mark.asyncio
async def test_real_entry_beats_anchor(llm_cache_with_anchors, hass):
    """Test that real cached entries take priority over anchors."""
    # Store a real command
    await llm_cache_with_anchors.store(
        text="Mach die Funzel im Hobbyraum an",
        intent="HassTurnOn",
        entity_ids=["light.hobby_funzel"],
        slots={"area": "Hobbyraum"},
        verified=True,
    )

    # Same command should now return the real entry
    result = await llm_cache_with_anchors.lookup("Mach die Funzel im Hobbyraum an")

    # Real entry should beat anchor because exact match scores higher
    assert result is not None
    assert result["entity_ids"] == ["light.hobby_funzel"]
    assert result["source"] == "learned"


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

    # Explicit anchor hit returns None
    assert result is None
    
    stats = llm_cache_with_anchors.get_stats()
    assert stats["anchor_escalations"] > 0


@pytest.mark.asyncio
async def test_off_hits_off_anchor_not_on(llm_cache_with_anchors, hass):
    """Test that 'aus' commands hit Off anchor even if 'an' is in cache."""
    # Store a TurnOn command
    await llm_cache_with_anchors.store(
        text="Schalte das Licht im Hobbyraum an",
        intent="HassTurnOn",
        entity_ids=["light.hobby_spots"],
        slots={"area": "Hobbyraum"},
        verified=True,
    )

    # 'aus' query should hit HassTurnOff anchor, NOT the HassTurnOn cached entry
    # Note: HassTurnOff anchor for 'Hobbyraum' might not exist if it's not in TEST_AREAS
    # So we use 'Küche' which IS in TEST_AREAS and HAS anchors.
    
    await llm_cache_with_anchors.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche_spots"],
        slots={"area": "Küche"},
        verified=True,
    )
    
    result = await llm_cache_with_anchors.lookup("Schalte das Licht in der Küche aus")

    assert result is None
    
    stats = llm_cache_with_anchors.get_stats()
    assert stats["anchor_escalations"] > 0
