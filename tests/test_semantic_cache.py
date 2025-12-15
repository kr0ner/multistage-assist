"""Tests for SemanticCacheCapability.

Tests semantic command caching with sentence embeddings.
"""

from unittest.mock import MagicMock
import pytest
import numpy as np

from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_embedding_model():
    """Mock the sentence transformer model."""
    mock_model = MagicMock()
    # Return random but consistent embeddings based on input hash
    def encode_fn(text, convert_to_numpy=True):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)
    mock_model.encode = encode_fn
    return mock_model


@pytest.fixture
def semantic_cache(hass, config_entry, mock_embedding_model):
    """Create semantic cache with mocked model."""
    cache = SemanticCacheCapability(hass, config_entry.data)
    cache._model = mock_embedding_model
    cache._loaded = True
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


async def test_cache_hit_similar_phrasing(semantic_cache, hass, mock_embedding_model):
    """Test that similar phrasing triggers cache hit."""
    # Override encode to return similar embeddings for similar phrases
    base_embedding = np.random.randn(384).astype(np.float32)
    
    def similar_encode(text, convert_to_numpy=True):
        if "küche" in text.lower() and ("licht" in text.lower() or "lampe" in text.lower()):
            # Return slightly perturbed base embedding for similar phrases
            noise = np.random.randn(384) * 0.05
            return (base_embedding + noise).astype(np.float32)
        return np.random.randn(384).astype(np.float32)
    
    mock_embedding_model.encode = similar_encode
    
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
    assert result["score"] > 0.9


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
    assert result is None or result["score"] < 0.92


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
