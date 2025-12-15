"""Tests for SemanticCacheCapability with Ollama embeddings.

Tests semantic command caching using mocked Ollama API.
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
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(1024).tolist()  # mxbai-embed-large uses 1024 dim
        return {"embedding": embedding}
    return create_response


@pytest.fixture
def semantic_cache(hass, config_entry, mock_ollama_response):
    """Create semantic cache with mocked Ollama API."""
    cache = SemanticCacheCapability(hass, config_entry.data)
    
    # Mock the _get_embedding method
    async def mock_get_embedding(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)
    
    cache._get_embedding = mock_get_embedding
    return cache


# ============================================================================
# TESTS
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


async def test_cache_hit_bypasses_llm(semantic_cache, hass):
    """Test that cache hit returns stored data without LLM."""
    # Store a command
    await semantic_cache.store(
        text="Licht in der Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )
    
    # Lookup same command
    result = await semantic_cache.lookup("Licht in der Küche an")
    
    assert result is not None
    assert result["intent"] == "HassTurnOn"
    assert result["entity_ids"] == ["light.kuche"]
    assert result["score"] > 0.99  # Same text should be near-identical


async def test_cache_hit_similar_phrasing(semantic_cache, hass):
    """Test that similar phrasing triggers cache hit."""
    # Override _get_embedding to return similar embeddings for similar phrases
    base_embedding = np.random.randn(1024).astype(np.float32)
    
    async def similar_get_embedding(text):
        if "küche" in text.lower() and ("licht" in text.lower() or "lampe" in text.lower()):
            # Return slightly perturbed base embedding for similar phrases
            np.random.seed(42)  # Same seed = same noise
            noise = np.random.randn(1024) * 0.05
            return (base_embedding + noise).astype(np.float32)
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(1024).astype(np.float32)
    
    semantic_cache._get_embedding = similar_get_embedding
    
    # Store command
    await semantic_cache.store(
        text="Licht Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={"area": "Küche"},
        verified=True,
    )
    
    # Lookup similar phrasing
    result = await semantic_cache.lookup("Mach das Licht in der Küche an")
    
    assert result is not None
    assert result["score"] > 0.8


async def test_cache_miss_different_intent(semantic_cache, hass):
    """Test that different intents don't match."""
    await semantic_cache.store(
        text="Licht Küche an",
        intent="HassTurnOn",
        entity_ids=["light.kuche"],
        slots={},
        verified=True,
    )
    
    # Lookup opposite intent - should miss due to different embedding
    result = await semantic_cache.lookup("Schalte alle Lichter aus")
    
    # Should be low similarity (different semantics)
    assert result is None or result["score"] < 0.85


async def test_cache_preserves_disambiguation(semantic_cache, hass):
    """Test that disambiguation info is preserved in cache."""
    await semantic_cache.store(
        text="Licht im Bad an",
        intent="HassTurnOn",
        entity_ids=["light.bad_spiegel"],  # After disambiguation
        slots={"area": "Bad"},
        required_disambiguation=True,
        disambiguation_options={"light.bad": "Badezimmer", "light.bad_spiegel": "Bad Spiegel"},
        verified=True,
    )
    
    result = await semantic_cache.lookup("Licht im Bad an")
    
    assert result is not None
    assert result["required_disambiguation"] is True
    assert "light.bad" in result["disambiguation_options"]


async def test_cache_not_stored_on_failure(semantic_cache, hass):
    """Test that unverified commands are not stored."""
    await semantic_cache.store(
        text="Fehlerhafte Aktion",
        intent="HassTurnOn",
        entity_ids=["light.missing"],
        slots={},
        verified=False,
    )
    
    assert len(semantic_cache._cache) == 0


async def test_cache_disabled(hass, config_entry):
    """Test that cache can be disabled via config."""
    config = dict(config_entry.data)
    config["cache_enabled"] = False
    
    cache = SemanticCacheCapability(hass, config)
    
    # Mock embedding
    async def mock_get_embedding(text):
        return np.random.randn(1024).astype(np.float32)
    cache._get_embedding = mock_get_embedding
    
    await cache.store(
        text="Test",
        intent="HassTurnOn",
        entity_ids=["light.test"],
        slots={},
        verified=True,
    )
    
    # Should not store when disabled
    assert len(cache._cache) == 0
    
    # Lookup should return None when disabled
    result = await cache.lookup("Test")
    assert result is None


async def test_cache_custom_config(hass, config_entry):
    """Test that cache uses custom config options."""
    config = dict(config_entry.data)
    config["embedding_ip"] = "192.168.178.2"
    config["embedding_port"] = 11434
    config["embedding_model"] = "nomic-embed-text"
    config["cache_similarity_threshold"] = 0.9
    
    cache = SemanticCacheCapability(hass, config)
    
    assert cache.embedding_ip == "192.168.178.2"
    assert cache.embedding_port == 11434
    assert cache.embedding_model == "nomic-embed-text"
    assert cache.threshold == 0.9
