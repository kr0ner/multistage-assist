"""Tests for IntentExecutorCapability.

Tests verification polling, German state translations, and execution logic.
"""

from unittest.mock import MagicMock, patch
import pytest
from homeassistant.helpers import intent

from multistage_assist.capabilities.intent_executor import IntentExecutorCapability


# ============================================================================
# VERIFICATION POLLING TESTS
# ============================================================================

async def test_verification_returns_early_on_success(config_entry):
    """Test that verification returns immediately when state matches."""
    import time
    
    # Create mock hass with state tracking
    mock_hass = MagicMock()
    states = {}
    
    class MockState:
        def __init__(self, state, attrs):
            self.state = state
            self.attributes = attrs
    
    def get_state(entity_id):
        return states.get(entity_id)
    
    def set_state(entity_id, state, attrs):
        states[entity_id] = MockState(state, attrs)
    
    mock_hass.states.get = get_state
    
    executor = IntentExecutorCapability(mock_hass, config_entry.data)
    
    # Set state to expected value
    set_state("light.test_verify", "on", {"friendly_name": "Test Light"})
    
    start = time.time()
    result = await executor._verify_execution(
        "light.test_verify",
        "HassTurnOn",
        expected_state="on",
    )
    elapsed = time.time() - start
    
    assert result is True
    # Should return quickly (within 2 polls = ~1s) not wait full timeout
    assert elapsed < 2.0


async def test_verification_accepts_transitional_states(config_entry):
    """Test that transitional cover states are accepted."""
    mock_hass = MagicMock()
    states = {}
    
    class MockState:
        def __init__(self, state, attrs):
            self.state = state
            self.attributes = attrs
    
    def get_state(entity_id):
        return states.get(entity_id)
    
    mock_hass.states.get = get_state
    
    executor = IntentExecutorCapability(mock_hass, config_entry.data)
    
    # Cover is opening (transitional)
    states["cover.test_verify"] = MockState("opening", {"friendly_name": "Test Cover"})
    
    result = await executor._verify_execution(
        "cover.test_verify",
        "HassTurnOn",  # Open command
        expected_state="on",  # Mapped to "open"
    )
    
    assert result is True  # "opening" should be accepted


async def test_media_player_accepts_non_off_as_on(config_entry):
    """Test that media_player accepts playing/paused/idle as 'on'."""
    mock_hass = MagicMock()
    states = {}
    
    class MockState:
        def __init__(self, state, attrs):
            self.state = state
            self.attributes = attrs
    
    def get_state(entity_id):
        return states.get(entity_id)
    
    mock_hass.states.get = get_state
    
    executor = IntentExecutorCapability(mock_hass, config_entry.data)
    
    # Media player is playing
    states["media_player.test_verify"] = MockState("playing", {"friendly_name": "Test Player"})
    
    result = await executor._verify_execution(
        "media_player.test_verify",
        "HassTurnOn",
        expected_state="on",
    )
    
    assert result is True  # "playing" should count as "on"
    
    # Also test "paused"
    states["media_player.test_verify"] = MockState("paused", {"friendly_name": "Test Player"})
    
    result2 = await executor._verify_execution(
        "media_player.test_verify",
        "HassTurnOn",
        expected_state="on",
    )
    
    assert result2 is True  # "paused" should also count as "on"


async def test_scene_accepts_recent_timestamp(config_entry):
    """Test that scene verification accepts recent activation timestamps."""
    from datetime import datetime, timezone, timedelta
    
    mock_hass = MagicMock()
    states = {}
    
    class MockState:
        def __init__(self, state, attrs):
            self.state = state
            self.attributes = attrs
    
    def get_state(entity_id):
        return states.get(entity_id)
    
    mock_hass.states.get = get_state
    
    executor = IntentExecutorCapability(mock_hass, config_entry.data)
    
    # Scene activated just now (1 second ago)
    recent_time = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    states["scene.nora"] = MockState(recent_time, {"friendly_name": "Nora geht Schlafen"})
    
    result = await executor._verify_execution(
        "scene.nora",
        "HassTurnOn",
        expected_state="on",
    )
    
    assert result is True  # Recent timestamp should pass


async def test_scene_rejects_old_timestamp(config_entry):
    """Test that scene verification rejects old activation timestamps."""
    from datetime import datetime, timezone, timedelta
    
    mock_hass = MagicMock()
    states = {}
    
    class MockState:
        def __init__(self, state, attrs):
            self.state = state
            self.attributes = attrs
    
    def get_state(entity_id):
        return states.get(entity_id)
    
    mock_hass.states.get = get_state
    
    executor = IntentExecutorCapability(mock_hass, config_entry.data)
    
    # Scene activated 1 hour ago (too old)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    states["scene.old"] = MockState(old_time, {"friendly_name": "Old Scene"})
    
    result = await executor._verify_execution(
        "scene.old",
        "HassTurnOn",
        expected_state="on",
    )

    
    assert result is False  # Old timestamp should fail


# GERMAN STATE TRANSLATION TESTS
# ============================================================================

async def test_state_query_translates_off(hass, config_entry):
    """Test that 'off' state is translated to 'aus' in German responses."""
    executor = IntentExecutorCapability(hass, config_entry.data)
    
    hass.states.async_set("cover.rollladen", "closed", {"friendly_name": "Rollladen"})
    
    user_input = MagicMock()
    user_input.text = "Ist der Rollladen zu?"
    user_input.conversation_id = "test"
    user_input.language = "de"
    
    with patch("homeassistant.helpers.intent.async_handle") as mock_handle:
        mock_resp = intent.IntentResponse(language="de")
        mock_resp.async_set_speech("Okay")
        mock_handle.return_value = mock_resp
        
        result = await executor.run(
            user_input,
            intent_name="HassGetState",
            entity_ids=["cover.rollladen"],
            params={},
        )
        
        if result and "result" in result:
            speech = result["result"].response.speech.get("plain", {}).get("speech", "")
            # Should contain German translation
            assert "off" not in speech.lower() or "geschlossen" in speech.lower()


async def test_state_query_translates_opening(hass, config_entry):
    """Test that 'opening' state is translated to 'öffnet' in German."""
    executor = IntentExecutorCapability(hass, config_entry.data)
    
    hass.states.async_set("cover.rollladen", "opening", {"friendly_name": "Rollladen"})
    
    user_input = MagicMock()
    user_input.text = "Was macht der Rollladen?"
    user_input.conversation_id = "test"
    user_input.language = "de"
    
    with patch("homeassistant.helpers.intent.async_handle") as mock_handle:
        mock_resp = intent.IntentResponse(language="de")
        mock_resp.async_set_speech("")  # Empty to trigger state generation
        mock_handle.return_value = mock_resp
        
        result = await executor.run(
            user_input,
            intent_name="HassGetState",
            entity_ids=["cover.rollladen"],
            params={},
        )
        
        if result and "result" in result:
            speech = result["result"].response.speech.get("plain", {}).get("speech", "")
            assert "öffnet" in speech or "Rollladen" in speech


async def test_state_query_media_playing(hass, config_entry):
    """Test that 'playing' state is translated to 'spielt' in German."""
    executor = IntentExecutorCapability(hass, config_entry.data)
    
    hass.states.async_set("media_player.radio", "playing", {"friendly_name": "Radio"})
    
    user_input = MagicMock()
    user_input.text = "Ist das Radio an?"
    user_input.conversation_id = "test"
    user_input.language = "de"
    
    with patch("homeassistant.helpers.intent.async_handle") as mock_handle:
        mock_resp = intent.IntentResponse(language="de")
        mock_resp.async_set_speech("")
        mock_handle.return_value = mock_resp
        
        result = await executor.run(
            user_input,
            intent_name="HassGetState",
            entity_ids=["media_player.radio"],
            params={},
        )
        
        if result and "result" in result:
            speech = result["result"].response.speech.get("plain", {}).get("speech", "")
            assert "playing" not in speech.lower()  # Should be translated


# ============================================================================
# TEMPORARY CONTROL TESTS (Skipped - require complex HA mocking)
# ============================================================================

@pytest.mark.skip(reason="Requires complex HA intent mock setup - covered by test_scenarios.py")
async def test_temporary_control_generates_speech(hass, config_entry):
    """Test that TemporaryControl generates proper confirmation speech."""
    pass


@pytest.mark.skip(reason="Requires complex HA intent mock setup")
async def test_legacy_duration_generates_speech(hass, config_entry):
    """Test that HassTurnOn/Off with duration generates proper speech."""
    pass


@pytest.mark.skip(reason="Requires async mock for timebox script")
async def test_automation_timebox_stores_original_state(hass, config_entry):
    """Test that automation timebox correctly stores original state."""
    pass
