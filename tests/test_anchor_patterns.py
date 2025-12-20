"""Unit tests for semantic anchor hit/miss detection with REAL reranker.

Tests that anchors correctly match (hit) and reject (miss) utterances.
Each intent has 10 test cases for hits and 10 for misses.

Requires: Reranker server running at localhost:9876
Start with: cd reranker-addon && RERANKER_MODEL=BAAI/bge-reranker-base python3 app.py
"""

import os
import pytest
import aiohttp

# Mark all tests as integration tests (requires external reranker)
# Run with: pytest -m integration tests/test_anchor_patterns.py
# Skip with: pytest -m "not integration"
pytestmark = pytest.mark.integration

# Reranker configuration
RERANKER_HOST = os.getenv("RERANKER_HOST", "192.168.178.2")
RERANKER_PORT = int(os.getenv("RERANKER_PORT", "9876"))

# Per-domain thresholds - optimized through systematic testing
DOMAIN_THRESHOLDS = {
    "light": 0.73,
    "switch": 0.73,
    "fan": 0.73,
    "cover": 0.73,
    "climate": 0.69,
}
DEFAULT_THRESHOLD = 0.73


# ============================================================================
# HELPER FUNCTION
# ============================================================================


async def call_reranker(query: str, candidates: list) -> dict:
    """Call reranker API and return scores."""
    url = f"http://{RERANKER_HOST}:{RERANKER_PORT}/rerank"
    payload = {"query": query, "candidates": candidates}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=30) as resp:
            return await resp.json()


async def check_reranker_available() -> bool:
    """Check if reranker is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{RERANKER_HOST}:{RERANKER_PORT}/health", 
                timeout=2
            ) as resp:
                return resp.status == 200
    except:
        return False


def pytest_configure(config):
    """Skip all tests if reranker not available (sync check at collection time)."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((RERANKER_HOST, RERANKER_PORT))
        if result != 0:
            pytest.skip(
                f"Reranker not available at {RERANKER_HOST}:{RERANKER_PORT}",
                allow_module_level=True
            )
    except:
        pytest.skip(
            f"Reranker not available at {RERANKER_HOST}:{RERANKER_PORT}",
            allow_module_level=True
        )
    finally:
        sock.close()


# ============================================================================
# TURN ON ANCHOR TESTS
# ============================================================================


class TestHassTurnOnAnchorHits:
    """Test cases that SHOULD match HassTurnOn anchors."""

    ANCHOR = "Schalte die Deckenlampe in der Küche an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Schalte die Deckenlampe in der Küche an",  # Exact
        "Mach die Deckenlampe in der Küche an",  # Synonym
        "Deckenlampe in der Küche einschalten",  # Reordered
        "Küche Deckenlampe an",  # Informal
        "Bitte schalte die Deckenlampe in der Küche an",  # Polite
        "Die Deckenlampe in der Küche anmachen",  # Variant
        "Mach mal die Deckenlampe in der Küche an",  # Casual
        "In der Küche die Deckenlampe anschalten",  # Reordered
        "Kannst du die Deckenlampe in der Küche anschalten",  # Question
        "Deckenlampe Küche anmachen",  # Short
    ])
    async def test_turn_on_hit(self, query):
        """Test that TurnOn synonyms score above threshold."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        threshold = DOMAIN_THRESHOLDS["light"]
        
        assert score >= threshold, \
            f"Expected HIT for '{query}' but got score {score:.3f} (threshold {threshold})"


class TestHassTurnOnAnchorMisses:
    """Test cases that should NOT match HassTurnOn anchors."""

    ANCHOR = "Schalte die Deckenlampe in der Küche an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schalte die Deckenlampe in der Küche aus", "opposite action"),
        ("Schalte die Deckenlampe im Wohnzimmer an", "different room"),
        ("Schalte den Fernseher in der Küche an", "different device"),
        ("Mach die Deckenlampe in der Küche heller", "different intent: brightness"),
        ("Dimme die Deckenlampe in der Küche", "dimming intent"),
        ("Ist die Deckenlampe in der Küche an?", "state query"),
        ("Wie wird das Wetter morgen?", "unrelated query"),
        ("Öffne die Rollos in der Küche", "different device type"),
        ("Schalte die Deckenlampe für 10 Minuten an", "temporal control"),
        ("Stelle die Heizung in der Küche auf 21 Grad", "climate control"),
    ])
    async def test_turn_on_miss(self, query, reason):
        """Test that non-matching queries score below threshold."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["light"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# TURN OFF ANCHOR TESTS
# ============================================================================


class TestHassTurnOffAnchorHits:
    """Test cases that SHOULD match HassTurnOff anchors."""

    ANCHOR = "Schalte das Licht im Bad aus"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Schalte das Licht im Bad aus",
        "Mach das Licht im Bad aus",
        "Licht im Bad ausschalten",
        "Bad Licht aus",
        "Bitte mach das Licht im Bad aus",
        "Das Licht im Bad ausmachen",
        "Im Bad das Licht ausschalten",
        "Licht aus im Bad",
        "Kannst du das Licht im Bad ausschalten",
        "Mach mal das Licht im Bad aus",
    ])
    async def test_turn_off_hit(self, query):
        """Test TurnOff anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["light"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassTurnOffAnchorMisses:
    """Test cases that should NOT match HassTurnOff anchors."""

    ANCHOR = "Schalte das Licht im Bad aus"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schalte das Licht im Bad an", "opposite: turn on"),
        ("Schalte das Licht im Schlafzimmer aus", "different room"),
        ("Schalte den Fernseher im Bad aus", "different device"),
        ("Mach das Licht im Bad heller", "brightness intent"),
        ("Ist das Licht im Bad an?", "state query"),
        ("Öffne die Rollos im Bad", "different device type"),
        ("Wie ist die Temperatur im Bad?", "climate query"),
        ("Schalte das Licht im Bad für 5 Minuten aus", "temporal"),
        ("Was ist das Wetter?", "unrelated"),
        ("Stelle einen Timer für 10 Minuten", "timer intent"),
    ])
    async def test_turn_off_miss(self, query, reason):
        """Test TurnOff anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["light"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# LIGHT SET (BRIGHTNESS) ANCHOR TESTS
# ============================================================================


class TestHassLightSetAnchorHits:
    """Test cases that SHOULD match HassLightSet anchors."""

    ANCHOR = "Mach das Licht im Wohnzimmer heller"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Mach das Licht im Wohnzimmer heller",
        "Wohnzimmer Licht heller",
        "Das Licht im Wohnzimmer heller machen",
        "Kannst du das Licht im Wohnzimmer heller machen",
        "Bitte mach das Licht im Wohnzimmer heller",
        "Licht heller im Wohnzimmer",
        "Mach mal das Licht im Wohnzimmer heller",
        "Im Wohnzimmer das Licht heller",
        "Wohnzimmer heller bitte",
        "Das Licht heller machen im Wohnzimmer",
    ])
    async def test_light_set_hit(self, query):
        """Test LightSet anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["light"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassLightSetAnchorMisses:
    """Test cases that should NOT match HassLightSet anchors."""

    ANCHOR = "Mach das Licht im Wohnzimmer heller"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Mach das Licht im Wohnzimmer dunkler", "opposite: dunkler"),
        ("Mach das Licht im Büro heller", "different room"),
        ("Schalte das Licht im Wohnzimmer an", "on/off instead"),
        ("Schalte das Licht im Wohnzimmer aus", "off intent"),
        ("Ist das Licht im Wohnzimmer an?", "state query"),
        ("Mach den Fernseher heller", "different device type"),
        ("Stelle die Heizung wärmer", "climate control"),
        ("Öffne die Rollos im Wohnzimmer", "cover intent"),
        ("Timer für 5 Minuten", "timer intent"),
        ("Was ist die Helligkeit?", "state query"),
    ])
    async def test_light_set_miss(self, query, reason):
        """Test LightSet anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["light"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# GET STATE ANCHOR TESTS
# ============================================================================


class TestHassGetStateAnchorHits:
    """Test cases that SHOULD match HassGetState anchors."""

    ANCHOR = "Ist das Licht im Flur an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Ist das Licht im Flur an",
        "Ist das Licht im Flur an?",
        "Ist die Lampe im Flur an",
        "Brennt das Licht im Flur",
        "Ist das Licht im Flur eingeschaltet",
        "Flur Licht an?",
        "Ist im Flur das Licht an",
        "Leuchtet das Licht im Flur",
        "Ist das Licht im Flur noch an",
        "Läuft das Licht im Flur",
    ])
    async def test_get_state_hit(self, query):
        """Test GetState anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["light"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassGetStateAnchorMisses:
    """Test cases that should NOT match HassGetState anchors."""

    ANCHOR = "Ist das Licht im Flur an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schalte das Licht im Flur an", "action instead of query"),
        ("Mach das Licht im Flur aus", "off action"),
        ("Ist das Licht im Bad an", "different room"),
        ("Mach das Licht im Flur heller", "brightness action"),
        ("Wie warm ist es im Flur", "temperature query"),
        ("Öffne die Rollos im Flur", "cover action"),
        ("Ist der Fernseher an", "different device"),
        ("Stelle einen Timer", "timer intent"),
        ("Wie wird das Wetter", "weather query"),
        ("Schalte das Licht im Flur für 5 Minuten an", "temporal"),
    ])
    async def test_get_state_miss(self, query, reason):
        """Test GetState anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["light"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# NOTE: TemporaryControl tests removed - duration patterns are now
# handled by keyword detection in semantic_cache.py, not semantic matching


# ============================================================================
# SET POSITION (COVER) ANCHOR TESTS
# ============================================================================


class TestHassSetPositionAnchorHits:
    """Test cases that SHOULD match HassSetPosition anchors."""

    ANCHOR = "Öffne die Rollos im Schlafzimmer"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Öffne die Rollos im Schlafzimmer",
        "Mach die Rollos im Schlafzimmer auf",
        "Rollos im Schlafzimmer öffnen",
        "Schlafzimmer Rollos auf",
        "Bitte öffne die Rollos im Schlafzimmer",
        "Die Rollos im Schlafzimmer aufmachen",
        "Fahr die Rollos im Schlafzimmer hoch",
        "Rollos hoch im Schlafzimmer",
        "Im Schlafzimmer die Rollos öffnen",
        "Kannst du die Rollos im Schlafzimmer öffnen",
    ])
    async def test_set_position_hit(self, query):
        """Test SetPosition anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["cover"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassSetPositionAnchorMisses:
    """Test cases that should NOT match HassSetPosition anchors."""

    ANCHOR = "Öffne die Rollos im Schlafzimmer"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schließe die Rollos im Schlafzimmer", "opposite: close"),
        ("Öffne die Rollos im Wohnzimmer", "different room"),
        ("Schalte das Licht im Schlafzimmer an", "light intent"),
        ("Öffne das Fenster im Schlafzimmer", "window not cover"),
        ("Öffne die Tür", "door not cover"),
        ("Ist das Rollo im Schlafzimmer offen", "state query"),
        ("Stelle die Heizung im Schlafzimmer ein", "climate intent"),
        ("Mach das Licht im Schlafzimmer an", "light on intent"),
        ("Wie ist das Wetter", "weather query"),
        ("Öffne die App", "different domain"),
    ])
    async def test_set_position_miss(self, query, reason):
        """Test SetPosition anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["cover"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


class TestHassSetPositionCloseAnchorHits:
    """Test cases that SHOULD match HassSetPosition close anchors."""

    ANCHOR = "Mach die Rollos im Schlafzimmer zu"  # Different phrasing for close

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Mach die Rollos im Schlafzimmer zu",
        "Schließe die Rollos im Schlafzimmer",
        "Rollos im Schlafzimmer schließen",
        "Schlafzimmer Rollos zu",
        "Bitte schließe die Rollos im Schlafzimmer",
        "Die Rollos im Schlafzimmer zumachen",
        "Fahr die Rollos im Schlafzimmer runter",
        "Rollos runter im Schlafzimmer",
        "Im Schlafzimmer die Rollos schließen",
        "Kannst du die Rollos im Schlafzimmer schließen",
    ])
    async def test_set_position_close_hit(self, query):
        """Test SetPosition close anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["cover"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassSetPositionCloseAnchorMisses:
    """Test cases that should NOT match HassSetPosition close anchors."""

    ANCHOR = "Mach die Rollos im Schlafzimmer zu"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Öffne die Rollos im Schlafzimmer", "opposite: open"),
        ("Schließe die Rollos im Wohnzimmer", "different room"),
        ("Schalte das Licht im Schlafzimmer aus", "light intent"),
        ("Ist das Rollo im Schlafzimmer zu", "state query"),
        ("Mach das Licht im Schlafzimmer aus", "light off intent"),
        ("Wie ist das Wetter", "weather query"),
    ])
    async def test_set_position_close_miss(self, query, reason):
        """Test SetPosition close anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["cover"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"
# CLIMATE SET TEMPERATURE ANCHOR TESTS
# ============================================================================


class TestHassClimateSetTemperatureAnchorHits:
    """Test cases that SHOULD match HassClimateSetTemperature anchors."""

    ANCHOR = "Stelle die Heizung in der Küche auf 21 Grad"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Stelle die Heizung in der Küche auf 21 Grad",
        "Heizung in der Küche auf 20 Grad",
        "Küche auf 22 Grad stellen",
        "Mach die Heizung in der Küche wärmer",
        "Stelle die Temperatur in der Küche auf 19 Grad",
        "Küche Heizung 21 Grad",
        "Die Heizung in der Küche auf 23 Grad einstellen",
        "Bitte stell die Heizung in der Küche auf 21",
        "In der Küche 20 Grad einstellen",
        "Heizung Küche wärmer",
    ])
    async def test_climate_set_temp_hit(self, query):
        """Test ClimateSetTemperature anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["climate"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestHassClimateSetTemperatureAnchorMisses:
    """Test cases that should NOT match HassClimateSetTemperature anchors."""

    ANCHOR = "Stelle die Heizung in der Küche auf 21 Grad"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Stelle die Heizung im Bad auf 21 Grad", "different room"),
        ("Wie warm ist es in der Küche", "temperature query"),
        ("Schalte das Licht in der Küche an", "light intent"),
        ("Schließe die Rollos in der Küche", "cover intent"),
        ("Schalte den Ventilator in der Küche an", "fan intent"),
        ("Wie ist die Luftqualität in der Küche", "air quality"),
        ("Stelle einen Timer für 21 Minuten", "timer intent"),
        ("Wie wird das Wetter morgen", "weather query"),
        ("Mach die Heizung in der Küche aus", "climate off"),
        ("Stelle die Lautstärke auf 21", "volume intent"),
    ])
    async def test_climate_set_temp_miss(self, query, reason):
        """Test ClimateSetTemperature anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["climate"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# SWITCH ANCHOR TESTS
# ============================================================================


class TestSwitchAnchorHits:
    """Test cases that SHOULD match switch anchors."""

    ANCHOR = "Schalte den Nachtmodus an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Schalte den Nachtmodus an",
        "Nachtmodus aktivieren",
        "Mach den Nachtmodus an",
        "Aktiviere den Nachtmodus",
        "Nachtmodus einschalten",
    ])
    async def test_switch_on_hit(self, query):
        """Test switch on anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DOMAIN_THRESHOLDS["switch"], \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestSwitchAnchorMisses:
    """Test cases that should NOT match switch anchors."""

    ANCHOR = "Schalte den Nachtmodus an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schalte den Nachtmodus aus", "opposite action"),
        ("Schalte das Licht an", "different domain"),
        ("Wie ist das Wetter", "unrelated"),
    ])
    async def test_switch_miss(self, query, reason):
        """Test switch anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DOMAIN_THRESHOLDS["switch"], \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# GLOBAL PATTERN TESTS (alle Lichter, etc.)
# ============================================================================


class TestGlobalPatternHits:
    """Test cases that SHOULD match global patterns."""

    ANCHOR_ALL_LIGHTS_OFF = "Schalte alle Lichter aus"
    ANCHOR_ALL_LIGHTS_ON = "Schalte alle Lichter an"
    ANCHOR_ALL_COVERS_DOWN = "Alle Rollos runter"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,anchor", [
        ("Alle Lichter aus", "Schalte alle Lichter aus"),
        ("Schalte alle Lichter aus", "Schalte alle Lichter aus"),
        ("Mach alle Lichter aus", "Schalte alle Lichter aus"),
        ("Alle Lampen ausschalten", "Schalte alle Lichter aus"),
        ("Alle Lichter an", "Schalte alle Lichter an"),
        ("Schalte alle Lichter an", "Schalte alle Lichter an"),
        ("Mach alle Lichter an", "Schalte alle Lichter an"),
        ("Alle Rollos runter", "Alle Rollos runter"),
        ("Fahre alle Rollos runter", "Alle Rollos runter"),
        ("Alle Rollläden schließen", "Alle Rollos runter"),
    ])
    async def test_global_pattern_hit(self, query, anchor):
        """Test global pattern anchor hits."""
        result = await call_reranker(query, [anchor])
        score = result["scores"][0]
        
        assert score >= DEFAULT_THRESHOLD, \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestGlobalPatternMisses:
    """Test cases that should NOT match global patterns."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,anchor,reason", [
        ("Schalte das Licht im Büro aus", "Schalte alle Lichter aus", "specific room"),
        ("Alle Lichter an", "Schalte alle Lichter aus", "opposite action"),
        ("Alle Rollos hoch", "Alle Rollos runter", "opposite action"),
        ("Schalte den Fernseher aus", "Schalte alle Lichter aus", "different domain"),
    ])
    async def test_global_pattern_miss(self, query, anchor, reason):
        """Test global pattern anchor misses."""
        result = await call_reranker(query, [anchor])
        score = result["scores"][0]
        
        assert score < DEFAULT_THRESHOLD, \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"


# ============================================================================
# MEDIA PLAYER ANCHOR TESTS
# ============================================================================


class TestMediaPlayerAnchorHits:
    """Test cases that SHOULD match media player anchors."""

    ANCHOR = "Schalte den Fernseher im Wohnzimmer an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "Schalte den Fernseher im Wohnzimmer an",
        "Fernseher im Wohnzimmer einschalten",
        "Mach den Fernseher im Wohnzimmer an",
        "TV im Wohnzimmer anmachen",
    ])
    async def test_media_player_hit(self, query):
        """Test media player anchor hits."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score >= DEFAULT_THRESHOLD, \
            f"Expected HIT for '{query}' but got score {score:.3f}"


class TestMediaPlayerAnchorMisses:
    """Test cases that should NOT match media player anchors."""

    ANCHOR = "Schalte den Fernseher im Wohnzimmer an"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,reason", [
        ("Schalte den Fernseher im Wohnzimmer aus", "opposite action"),
        ("Schalte den Fernseher in der Küche an", "different room"),
        ("Schalte das Licht im Wohnzimmer an", "different domain"),
        ("Pausiere den Fernseher", "different intent"),
    ])
    async def test_media_player_miss(self, query, reason):
        """Test media player anchor misses."""
        result = await call_reranker(query, [self.ANCHOR])
        score = result["scores"][0]
        
        assert score < DEFAULT_THRESHOLD, \
            f"Expected MISS for '{query}' ({reason}) but got score {score:.3f}"
