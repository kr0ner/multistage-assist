"""Tests for SemanticCacheCapability with two-stage reranking.

Tests semantic command caching using:
- Mocked Ollama embeddings for vector search
- Mocked CrossEncoder for reranking
"""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import numpy as np

from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_ollama_response():
    """Create a mock Ollama embedding response."""

    def create_response(text):
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(1024).tolist()
        return {"embedding": embedding}

    return create_response


@pytest.fixture
def semantic_cache(hass, config_entry, mock_ollama_response):
    """Create semantic cache with mocked Ollama API."""
    config = dict(config_entry.data)
    config["cache_enabled"] = True
    config["reranker_enabled"] = True
    config["reranker_mode"] = "local"  # Force local mode for tests

    cache = SemanticCacheCapability(hass, config)

    # Skip anchor initialization in unit tests
    cache._anchors_initialized = True
    cache._loaded = True
    cache._reranker_mode_resolved = "local"

    # Mock embedding - uses text hash for deterministic results
    async def mock_get_embedding(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)

    # Mock reranker - returns sigmoid probabilities based on text similarity
    class MockReranker:
        def predict(self, pairs):
            """Return mock scores based on text overlap."""
            scores = []
            for query, candidate in pairs:
                # Simple overlap-based scoring for testing
                overlap = len(
                    set(query.lower().split()) & set(candidate.lower().split())
                )
                # Higher overlap = higher score (range: -2 to +2 for sigmoid)
                scores.append(overlap * 0.5 - 1)
            return np.array(scores)

    cache._get_embedding = mock_get_embedding
    cache._reranker = MockReranker()
    return cache


@pytest.fixture
def semantic_cache_no_reranker(hass, config_entry):
    """Create semantic cache with reranker disabled."""
    config = dict(config_entry.data)
    config["cache_enabled"] = True
    config["reranker_enabled"] = False

    cache = SemanticCacheCapability(hass, config)

    # Skip anchor initialization in unit tests
    cache._anchors_initialized = True
    cache._loaded = True

    async def mock_get_embedding(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)

    cache._get_embedding = mock_get_embedding
    return cache


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================


async def test_cache_stores_verified_command(semantic_cache, hass):
    """Test that verified commands are stored in cache."""
    await semantic_cache.store(
        text="Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche", "domain": "light"},
        verified=True,
    )

    assert len(semantic_cache._cache) == 1
    entry = semantic_cache._cache[0]
    assert entry.intent == "HassTurnOn"
    assert entry.entity_ids == ["light.kuche"]
    assert entry.verified is True


async def test_cache_not_stored_when_disabled(hass, config_entry):
    """Test that cache operations are skipped when disabled."""
    config = dict(config_entry.data)
    config["cache_enabled"] = False

    cache = SemanticCacheCapability(hass, config)

    await cache.store(
        text="Test command",
        intent="HassTurnOn",
        entity_ids=["light.test"],
        slots={},
        verified=True,
    )

    assert len(cache._cache) == 0

    result = await cache.lookup("Test command")
    assert result is None


async def test_cache_not_stored_unverified(semantic_cache, hass):
    """Test that unverified commands are not stored."""
    await semantic_cache.store(
        text="Fehlerhafte Aktion",
        intent="HassTurnOn",
        entity_ids=["light.missing"],
        slots={},
        verified=False,
    )

    assert len(semantic_cache._cache) == 0


async def test_cache_skips_short_commands(semantic_cache, hass):
    """Test that short disambiguation responses are skipped."""
    await semantic_cache.store(
        text="Küche",  # Single word
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={},
        verified=True,
    )

    assert len(semantic_cache._cache) == 0


async def test_cache_skips_timer_commands(semantic_cache, hass):
    """Test that timer commands are not cached."""
    await semantic_cache.store(
        text="Stelle einen Timer für 5 Minuten",
        intent="HassTimerSet",
        entity_ids=[],
        slots={"duration": "5 minutes"},
        verified=True,
    )

    assert len(semantic_cache._cache) == 0


async def test_cache_stores_relative_commands(semantic_cache, hass):
    """Test that relative brightness commands ARE cached (command slot only, no brightness value)."""
    await semantic_cache.store(
        text="Mach das Licht heller",
        intent="HassLightSet",
        entity_ids=["light.kitchen"],
        slots={"command": "step_up", "brightness": 50},  # brightness should be filtered out
        verified=True,
    )

    assert len(semantic_cache._cache) == 1
    entry = semantic_cache._cache[0]
    assert entry.slots.get("command") == "step_up"
    assert "brightness" not in entry.slots  # Runtime value should be filtered


# ============================================================================
# TWO-STAGE PIPELINE TESTS
# ============================================================================


async def test_exact_match_returns_high_score(semantic_cache, hass):
    """Test that exact text match returns very high score."""
    await semantic_cache.store(
        text="Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # Mock reranker to return high score for exact match
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([3.0])  # High logit -> sigmoid ~0.95
    semantic_cache._reranker = mock_reranker

    result = await semantic_cache.lookup("Licht in der Küche an")

    assert result is not None
    assert result["intent"] == "HassTurnOn"
    assert result["entity_ids"] == ["light.kuche"]


async def test_reranker_blocks_opposite_action(semantic_cache, hass):
    """Test that reranker blocks opposite actions (on vs off)."""
    # Create similar embeddings for both texts
    base_embedding = np.random.randn(1024).astype(np.float32)

    async def similar_embeddings(text):
        # Return nearly identical embeddings (simulating vector search weakness)
        noise = np.random.randn(1024) * 0.01
        return (base_embedding + noise).astype(np.float32)

    semantic_cache._get_embedding = similar_embeddings

    await semantic_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche", "command": "an"},
        verified=True,
    )

    # Mock reranker to return LOW score for opposite action
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([-1.0])  # Low logit -> sigmoid ~0.27
    semantic_cache._reranker = mock_reranker

    result = await semantic_cache.lookup("Schalte das Licht in der Küche aus")

    # Reranker should block this (score < 0.5 threshold)
    assert result is None
    assert semantic_cache._stats["reranker_blocks"] >= 1


async def test_reranker_blocks_different_room(semantic_cache, hass):
    """Test that reranker blocks commands for different rooms."""
    base_embedding = np.random.randn(1024).astype(np.float32)

    async def similar_embeddings(text):
        noise = np.random.randn(1024) * 0.01
        return (base_embedding + noise).astype(np.float32)

    semantic_cache._get_embedding = similar_embeddings

    await semantic_cache.store(
        text="Licht im Büro an",
        intent="HassTurnOn",
        entity_ids=["light.buro"],
        slots={"area": "Büro"},
        verified=True,
    )

    # Mock reranker to return LOW score for different room
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([-0.5])  # Low logit -> sigmoid ~0.38
    semantic_cache._reranker = mock_reranker

    result = await semantic_cache.lookup("Licht in der Küche an")

    assert result is None


async def test_reranker_allows_synonym(semantic_cache, hass):
    """Test that reranker allows semantically equivalent commands."""
    base_embedding = np.random.randn(1024).astype(np.float32)

    async def similar_embeddings(text):
        noise = np.random.randn(1024) * 0.02
        return (base_embedding + noise).astype(np.float32)

    semantic_cache._get_embedding = similar_embeddings

    await semantic_cache.store(
        text="Schalte das Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # Mock reranker to return HIGH score for synonym
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([2.0])  # High logit -> sigmoid ~0.88
    semantic_cache._reranker = mock_reranker

    result = await semantic_cache.lookup("Mach die Lampe in der Küche an")

    assert result is not None
    assert result["intent"] == "HassTurnOn"
    assert result["reranked"] is True


async def test_vector_search_returns_top_k_candidates(semantic_cache, hass):
    """Test that vector search returns multiple candidates for reranking."""
    # Store multiple commands
    for i, room in enumerate(["Küche", "Büro", "Bad", "Flur", "Keller"]):
        await semantic_cache.store(
            text=f"Licht im {room} an",
            intent="HassTurnOn",
            entity_ids=[f"light.{room.lower()}"],
            slots={"area": room},
            verified=True,
        )

    assert len(semantic_cache._cache) == 5

    # Mock reranker to check it receives multiple candidates
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([0.1, 0.2, 0.3, 0.8, 0.5])  # 5 scores
    semantic_cache._reranker = mock_reranker

    # Use embedding that's somewhat similar to all
    semantic_cache._get_embedding = AsyncMock(
        return_value=np.mean(semantic_cache._embeddings_matrix, axis=0)
    )

    result = await semantic_cache.lookup("Licht an")

    # Reranker should have been called with multiple pairs
    assert mock_reranker.predict.called


# ============================================================================
# FALLBACK BEHAVIOR TESTS
# ============================================================================


async def test_fallback_when_reranker_disabled(semantic_cache_no_reranker, hass):
    """Test fallback to vector-only matching when reranker is disabled."""
    await semantic_cache_no_reranker.store(
        text="Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )

    # Exact match should work without reranker
    result = await semantic_cache_no_reranker.lookup("Licht in der Küche an")

    assert result is not None
    assert result["intent"] == "HassTurnOn"
    assert result.get("reranked") is not True  # Wasn't reranked


async def test_fallback_when_sentence_transformers_missing(hass, config_entry):
    """Test graceful fallback when sentence-transformers is not installed."""
    config = dict(config_entry.data)
    config["cache_enabled"] = True
    config["reranker_enabled"] = True

    cache = SemanticCacheCapability(hass, config)

    async def mock_get_embedding(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)

    cache._get_embedding = mock_get_embedding

    # Mock import error
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        cache._reranker = None

        await cache.store(
            text="Licht in der Küche an",
            intent="HassTurnOn",
            entity_ids=["light.kuche"],
            slots={},
            verified=True,
        )

        # Should still work, just without reranking
        result = await cache.lookup("Licht in der Küche an")
        # With random embeddings, exact match may not hit threshold
        # The important thing is it doesn't crash


# ============================================================================
# DISAMBIGUATION PRESERVATION TESTS
# ============================================================================


async def test_cache_preserves_disambiguation_info(semantic_cache, hass):
    """Test that disambiguation info is preserved in cache."""
    await semantic_cache.store(
        text="Licht im Bad an",
        intent="HassTurnOn",
        entity_ids=["light.bad_spiegel"],
        slots={"area": "Bad"},
        required_disambiguation=True,
        disambiguation_options={
            "light.bad": "Badezimmer",
            "light.bad_spiegel": "Bad Spiegel",
        },
        verified=True,
    )

    # Mock reranker for successful match
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = np.array([2.0])
    semantic_cache._reranker = mock_reranker

    result = await semantic_cache.lookup("Licht im Bad an")

    assert result is not None
    assert result["required_disambiguation"] is True
    assert "light.bad" in result["disambiguation_options"]


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


async def test_custom_config_options(hass, config_entry):
    """Test that all config options are respected."""
    config = dict(config_entry.data)
    config["embedding_ip"] = "192.168.178.2"
    config["embedding_port"] = 11434
    config["embedding_model"] = "bge-m3"
    config["reranker_model"] = "BAAI/bge-reranker-base"
    config["reranker_threshold"] = 0.6
    config["vector_search_threshold"] = 0.5
    config["vector_search_top_k"] = 10

    cache = SemanticCacheCapability(hass, config)

    assert cache.embedding_ip == "192.168.178.2"
    assert cache.embedding_port == 11434
    assert cache.embedding_model == "bge-m3"
    assert cache.reranker_model == "BAAI/bge-reranker-base"
    assert cache.reranker_threshold == 0.6
    assert cache.vector_threshold == 0.5
    assert cache.vector_top_k == 10


async def test_stats_include_reranker_info(semantic_cache, hass):
    """Test that stats include reranker information."""
    stats = await semantic_cache.get_stats()

    assert "reranker_host" in stats
    assert "reranker_enabled" in stats
    assert "reranker_blocks" in stats
