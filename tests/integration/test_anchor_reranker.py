"""Integration tests for semantic anchor hit/miss detection with REAL reranker.

These tests call the actual reranker API to verify semantic matching quality.
Run with: pytest tests/integration/test_anchor_reranker.py -v -m integration

Requires:
- Reranker addon running at local-semantic-reranker:9876 (or configure below)
- Ollama running with bge-m3 model for embeddings
"""

import os
import pytest
import numpy as np
import aiohttp

from multistage_assist.capabilities.semantic_cache import SemanticCacheCapability
from . import get_llm_config, OLLAMA_HOST, OLLAMA_PORT

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

# Reranker configuration
RERANKER_HOST = os.getenv("RERANKER_HOST", "192.168.178.108")  # Adjust for your setup
RERANKER_PORT = int(os.getenv("RERANKER_PORT", "9876"))
RERANKER_THRESHOLD = 0.60  # For bge-reranker-base


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def llm_cache(hass):
    """Create semantic cache with real Ollama embeddings and API reranker."""
    config = get_llm_config()
    config["cache_enabled"] = True
    config["embedding_model"] = "bge-m3"
    config["reranker_enabled"] = True
    config["reranker_mode"] = "api"  # Force API mode
    config["reranker_ip"] = RERANKER_HOST
    config["reranker_port"] = RERANKER_PORT
    config["reranker_threshold"] = RERANKER_THRESHOLD
    config["vector_search_threshold"] = 0.4
    config["vector_search_top_k"] = 10
    config["embedding_ip"] = OLLAMA_HOST
    config["embedding_port"] = OLLAMA_PORT

    cache = SemanticCacheCapability(hass, config)
    cache._anchors_initialized = True
    cache._loaded = True

    return cache


async def call_reranker(query: str, candidates: list[str]) -> dict:
    """Call reranker API directly for testing."""
    url = f"http://{RERANKER_HOST}:{RERANKER_PORT}/rerank"
    payload = {"query": query, "candidates": candidates}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=30) as resp:
            return await resp.json()


# ============================================================================
# TURN ON ANCHOR TESTS
# ============================================================================


class TestHassTurnOnRealReranker:
    """Test TurnOn anchor matching with real reranker."""
    
    ANCHOR = "Schalte die Deckenlampe in der Küche an"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS - should score above threshold
        ("Schalte die Deckenlampe in der Küche an", True),  # Exact
        ("Mach die Deckenlampe in der Küche an", True),  # Synonym
        ("Deckenlampe in der Küche einschalten", True),  # Reordered
        ("Küche Deckenlampe an", True),  # Informal
        ("Bitte schalte die Deckenlampe in der Küche an", True),  # Polite
        ("Die Deckenlampe in der Küche anmachen", True),  # Variant
        ("Mach mal die Deckenlampe in der Küche an", True),  # Casual
        ("In der Küche die Deckenlampe anschalten", True),  # Reordered
        ("Kannst du die Deckenlampe in der Küche anschalten", True),  # Question form
        ("Deckenlampe Küche anmachen", True),  # Short form
        
        # MISSES - should score below threshold
        ("Schalte die Deckenlampe in der Küche aus", False),  # Opposite: off
        ("Schalte die Deckenlampe im Wohnzimmer an", False),  # Different room
        ("Schalte den Fernseher in der Küche an", False),  # Different device
        ("Mach die Deckenlampe in der Küche heller", False),  # Brightness
        ("Dimme die Deckenlampe in der Küche", False),  # Dimming
        ("Ist die Deckenlampe in der Küche an?", False),  # State query
        ("Wie wird das Wetter morgen?", False),  # Unrelated
        ("Öffne die Rollos in der Küche", False),  # Cover
        ("Schalte die Deckenlampe für 10 Minuten an", False),  # Temporal
        ("Stelle die Heizung in der Küche auf 21 Grad", False),  # Climate
    ])
    async def test_turn_on_reranker(self, query, expected_hit):
        """Test reranker scoring for TurnOn intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestHassTurnOffRealReranker:
    """Test TurnOff anchor matching with real reranker."""
    
    ANCHOR = "Schalte das Licht im Bad aus"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Schalte das Licht im Bad aus", True),
        ("Mach das Licht im Bad aus", True),
        ("Licht im Bad ausschalten", True),
        ("Bad Licht aus", True),
        ("Bitte mach das Licht im Bad aus", True),
        ("Das Licht im Bad ausmachen", True),
        ("Im Bad das Licht ausschalten", True),
        ("Licht aus im Bad", True),
        ("Kannst du das Licht im Bad ausschalten", True),
        ("Mach mal das Licht im Bad aus", True),
        
        # MISSES
        ("Schalte das Licht im Bad an", False),  # Opposite
        ("Schalte das Licht im Schlafzimmer aus", False),  # Different room
        ("Schalte den Fernseher im Bad aus", False),  # Different device
        ("Mach das Licht im Bad heller", False),  # Brightness
        ("Ist das Licht im Bad an?", False),  # State query
        ("Öffne die Rollos im Bad", False),  # Cover
        ("Wie ist die Temperatur im Bad?", False),  # Climate query
        ("Schalte das Licht im Bad für 5 Minuten aus", False),  # Temporal
        ("Was ist das Wetter?", False),  # Unrelated
        ("Stelle einen Timer für 10 Minuten", False),  # Timer
    ])
    async def test_turn_off_reranker(self, query, expected_hit):
        """Test reranker scoring for TurnOff intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestHassLightSetRealReranker:
    """Test LightSet anchor matching with real reranker."""
    
    ANCHOR = "Mach das Licht im Wohnzimmer heller"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Mach das Licht im Wohnzimmer heller", True),
        ("Wohnzimmer Licht heller", True),
        ("Das Licht im Wohnzimmer heller machen", True),
        ("Kannst du das Licht im Wohnzimmer heller machen", True),
        ("Bitte mach das Licht im Wohnzimmer heller", True),
        ("Licht heller im Wohnzimmer", True),
        ("Mach mal das Licht im Wohnzimmer heller", True),
        ("Im Wohnzimmer das Licht heller", True),
        ("Wohnzimmer heller bitte", True),
        ("Das Licht heller machen im Wohnzimmer", True),
        
        # MISSES
        ("Mach das Licht im Wohnzimmer dunkler", False),  # Opposite
        ("Mach das Licht im Büro heller", False),  # Different room
        ("Schalte das Licht im Wohnzimmer an", False),  # On/off
        ("Schalte das Licht im Wohnzimmer aus", False),  # Off
        ("Ist das Licht im Wohnzimmer an?", False),  # State
        ("Mach den Fernseher heller", False),  # Different device
        ("Stelle die Heizung wärmer", False),  # Climate
        ("Öffne die Rollos im Wohnzimmer", False),  # Cover
        ("Timer für 5 Minuten", False),  # Timer
        ("Was ist die Helligkeit?", False),  # Query
    ])
    async def test_light_set_reranker(self, query, expected_hit):
        """Test reranker scoring for LightSet intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestHassGetStateRealReranker:
    """Test GetState anchor matching with real reranker."""
    
    ANCHOR = "Ist das Licht im Flur an"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Ist das Licht im Flur an", True),
        ("Ist das Licht im Flur an?", True),
        ("Ist die Lampe im Flur an", True),
        ("Brennt das Licht im Flur", True),
        ("Ist das Licht im Flur eingeschaltet", True),
        ("Flur Licht an?", True),
        ("Ist im Flur das Licht an", True),
        ("Leuchtet das Licht im Flur", True),
        ("Ist das Licht im Flur noch an", True),
        ("Läuft das Licht im Flur", True),
        
        # MISSES
        ("Schalte das Licht im Flur an", False),  # Action
        ("Mach das Licht im Flur aus", False),  # Off
        ("Ist das Licht im Bad an", False),  # Different room
        ("Mach das Licht im Flur heller", False),  # Brightness
        ("Wie warm ist es im Flur", False),  # Temperature
        ("Öffne die Rollos im Flur", False),  # Cover
        ("Ist der Fernseher an", False),  # Different device
        ("Stelle einen Timer", False),  # Timer
        ("Wie wird das Wetter", False),  # Weather
        ("Schalte das Licht im Flur für 5 Minuten an", False),  # Temporal
    ])
    async def test_get_state_reranker(self, query, expected_hit):
        """Test reranker scoring for GetState intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestTemporaryControlRealReranker:
    """Test TemporaryControl anchor matching with real reranker."""
    
    ANCHOR = "Schalte das Licht im Büro für eine Zeit an"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Schalte das Licht im Büro für 10 Minuten an", True),
        ("Mach das Licht im Büro für 5 Minuten an", True),
        ("Licht im Büro für eine Stunde an", True),
        ("Büro Licht für 30 Minuten", True),
        ("Schalte das Licht im Büro für kurze Zeit an", True),
        ("Mach das Licht im Büro temporär an", True),
        ("Licht im Büro für ne Weile an", True),
        ("Schalte das Licht im Büro vorübergehend an", True),
        ("Büro Licht für 2 Stunden anschalten", True),
        ("Mach mal das Licht im Büro für 15 Minuten an", True),
        
        # MISSES
        ("Schalte das Licht im Büro an", False),  # No duration
        ("Schalte das Licht im Büro aus", False),  # Off
        ("Schalte das Licht im Wohnzimmer für 10 Minuten an", False),  # Different room
        ("Mach das Licht im Büro heller", False),  # Brightness
        ("Ist das Licht im Büro an?", False),  # State
        ("Stelle einen Timer für 10 Minuten", False),  # Timer
        ("Öffne die Rollos im Büro", False),  # Cover
        ("Wie spät ist es", False),  # Time
        ("Erinnere mich in 10 Minuten", False),  # Reminder
        ("Schalte den Fernseher für 10 Minuten an", False),  # Different device
    ])
    async def test_temporary_control_reranker(self, query, expected_hit):
        """Test reranker scoring for TemporaryControl intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestHassSetPositionRealReranker:
    """Test SetPosition anchor matching with real reranker."""
    
    ANCHOR = "Öffne die Rollos im Schlafzimmer"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Öffne die Rollos im Schlafzimmer", True),
        ("Mach die Rollos im Schlafzimmer auf", True),
        ("Rollos im Schlafzimmer öffnen", True),
        ("Schlafzimmer Rollos auf", True),
        ("Bitte öffne die Rollos im Schlafzimmer", True),
        ("Die Rollos im Schlafzimmer aufmachen", True),
        ("Fahr die Rollos im Schlafzimmer hoch", True),
        ("Rollos hoch im Schlafzimmer", True),
        ("Im Schlafzimmer die Rollos öffnen", True),
        ("Kannst du die Rollos im Schlafzimmer öffnen", True),
        
        # MISSES
        ("Schließe die Rollos im Schlafzimmer", False),  # Opposite
        ("Öffne die Rollos im Wohnzimmer", False),  # Different room
        ("Schalte das Licht im Schlafzimmer an", False),  # Light
        ("Öffne das Fenster im Schlafzimmer", False),  # Window
        ("Öffne die Tür", False),  # Door
        ("Ist das Rollo im Schlafzimmer offen", False),  # State
        ("Stelle die Heizung im Schlafzimmer ein", False),  # Climate
        ("Mach das Licht im Schlafzimmer an", False),  # Light
        ("Wie ist das Wetter", False),  # Weather
        ("Öffne die App", False),  # App
    ])
    async def test_set_position_reranker(self, query, expected_hit):
        """Test reranker scoring for SetPosition intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"


class TestHassClimateSetTemperatureRealReranker:
    """Test ClimateSetTemperature anchor matching with real reranker."""
    
    ANCHOR = "Stelle die Heizung in der Küche auf 21 Grad"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_hit", [
        # HITS
        ("Stelle die Heizung in der Küche auf 21 Grad", True),
        ("Heizung in der Küche auf 20 Grad", True),
        ("Küche auf 22 Grad stellen", True),
        ("Mach die Heizung in der Küche wärmer", True),
        ("Stelle die Temperatur in der Küche auf 19 Grad", True),
        ("Küche Heizung 21 Grad", True),
        ("Die Heizung in der Küche auf 23 Grad einstellen", True),
        ("Bitte stell die Heizung in der Küche auf 21", True),
        ("In der Küche 20 Grad einstellen", True),
        ("Heizung Küche wärmer", True),
        
        # MISSES
        ("Stelle die Heizung im Bad auf 21 Grad", False),  # Different room
        ("Wie warm ist es in der Küche", False),  # Query
        ("Schalte das Licht in der Küche an", False),  # Light
        ("Schließe die Rollos in der Küche", False),  # Cover
        ("Schalte den Ventilator in der Küche an", False),  # Fan
        ("Wie ist die Luftqualität in der Küche", False),  # Air
        ("Stelle einen Timer für 21 Minuten", False),  # Timer
        ("Wie wird das Wetter morgen", False),  # Weather
        ("Mach die Heizung in der Küche aus", False),  # Climate off
        ("Stelle die Lautstärke auf 21", False),  # Volume
    ])
    async def test_climate_set_temp_reranker(self, query, expected_hit):
        """Test reranker scoring for ClimateSetTemperature intent."""
        result = await call_reranker(query, [self.ANCHOR])
        
        score = result["scores"][0]
        is_hit = score >= RERANKER_THRESHOLD
        
        assert is_hit == expected_hit, \
            f"Query: '{query}' got score {score:.3f}, expected {'hit' if expected_hit else 'miss'}"
