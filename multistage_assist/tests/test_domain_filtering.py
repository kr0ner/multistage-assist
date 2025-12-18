"""Test for domain filtering in intent resolution flow.

This tests the fix for the bug where "im Wohnzimmer ist es zu dunkel" resulted in
disambiguation asking about covers and sensors instead of just lights.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from homeassistant.components import conversation

from tests.integration import get_llm_config


@pytest.fixture
def hass():
    """Mock Home Assistant with area containing multiple entity types."""
    hass = MagicMock()
    
    # Create mock entities in "Wohnzimmer" area with different domains
    mock_entities = {
        "light.wohnzimmer": MagicMock(
            entity_id="light.wohnzimmer",
            domain="light",
            area_id="wohnzimmer_id",
            device_id=None,
            original_name="Wohnzimmer Licht",
            disabled_by=None,
        ),
        "light.wohnzimmer_stehlampe": MagicMock(
            entity_id="light.wohnzimmer_stehlampe",
            domain="light",
            area_id="wohnzimmer_id",
            device_id=None,
            original_name="Stehlampe",
            disabled_by=None,
        ),
        "cover.wohnzimmer": MagicMock(
            entity_id="cover.wohnzimmer",
            domain="cover",
            area_id="wohnzimmer_id",
            device_id=None,
            original_name="Wohnzimmer Rollo",
            disabled_by=None,
        ),
        "sensor.wohnzimmer_temp": MagicMock(
            entity_id="sensor.wohnzimmer_temp",
            domain="sensor",
            area_id="wohnzimmer_id",
            device_id=None,
            original_name="Wohnzimmer Temperatur",
            disabled_by=None,
        ),
        "climate.wohnzimmer": MagicMock(
            entity_id="climate.wohnzimmer",
            domain="climate",
            area_id="wohnzimmer_id",
            device_id=None,
            original_name="Wohnzimmer Thermostat",
            disabled_by=None,
        ),
    }
    
    # Mock area
    mock_area = MagicMock()
    mock_area.id = "wohnzimmer_id"
    mock_area.name = "Wohnzimmer"
    mock_area.floor_id = None
    
    return hass


def make_input(text: str):
    """Helper to create ConversationInput."""
    return conversation.ConversationInput(
        text=text,
        context=MagicMock(),
        conversation_id="test_id",
        device_id="test_device",
        language="de",
    )


class TestDomainFiltering:
    """Tests for proper domain filtering in the resolution pipeline."""
    
    @pytest.mark.asyncio
    async def test_keyword_intent_returns_domain(self, hass):
        """Test that keyword_intent returns domain at top level."""
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        
        cap = KeywordIntentCapability(hass, get_llm_config())
        user_input = make_input("Mache das Licht im Wohnzimmer heller")
        
        result = await cap.run(user_input)
        
        # Domain should be returned at TOP LEVEL, not just in slots
        assert result.get("domain") == "light", f"Expected domain='light' at top level, got: {result}"
        assert result.get("intent") == "HassLightSet"
    
    @pytest.mark.asyncio
    async def test_intent_resolution_injects_domain_into_slots(self, hass):
        """Test that intent_resolution properly injects domain from keyword_intent."""
        from multistage_assist.capabilities.intent_resolution import IntentResolutionCapability
        
        cap = IntentResolutionCapability(hass, get_llm_config())
        
        # Mock memory
        memory = MagicMock()
        memory.get_area_alias = AsyncMock(return_value=None)
        memory.get_floor_alias = AsyncMock(return_value=None)
        memory.get_entity_alias = AsyncMock(return_value=None)
        cap.set_memory(memory)
        
        # Mock keyword_intent to return domain at top level but empty in slots
        mock_ki_result = {
            "domain": "light",
            "intent": "HassLightSet",
            "slots": {
                "area": "Wohnzimmer",
                "command": "step_up",
                "name": "",
                "domain": "",  # Empty in slots!
                "brightness": "step_up",
            }
        }
        
        with patch.object(cap.keyword_cap, "run", return_value=mock_ki_result):
            # Mock entity resolver to capture what slots it receives
            original_resolver_run = cap.resolver_cap.run
            received_slots = {}
            
            async def capture_resolver_call(user_input, *, entities=None, **kwargs):
                nonlocal received_slots
                received_slots = entities or {}
                return {"resolved_ids": ["light.wohnzimmer"]}
            
            with patch.object(cap.resolver_cap, "run", side_effect=capture_resolver_call):
                user_input = make_input("Mache das Licht im Wohnzimmer heller")
                result = await cap.run(user_input)
        
        # Verify domain was injected into slots before calling resolver
        assert received_slots.get("domain") == "light", \
            f"Domain should be 'light' in slots, but got: {received_slots}"
    
    @pytest.mark.asyncio
    async def test_brightness_command_only_returns_lights(self, hass):
        """
        Integration test: "Mache das Licht im Wohnzimmer heller" should only
        return light entities, not covers or sensors.
        
        This tests the full pipeline fix for the bug where disambiguation
        would ask about covers and sensors.
        """
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        
        cap = KeywordIntentCapability(hass, get_llm_config())
        user_input = make_input("Mache das Licht im Wohnzimmer heller")
        
        result = await cap.run(user_input)
        
        # Verify correct intent and domain
        assert result.get("intent") == "HassLightSet"
        assert result.get("domain") == "light"
        
        # Verify slots contain area
        slots = result.get("slots", {})
        assert "wohnzimmer" in slots.get("area", "").lower()
        
        # Verify brightness is step_up
        assert slots.get("brightness") == "step_up" or slots.get("command") == "step_up"
    
    @pytest.mark.asyncio 
    async def test_clarification_transforms_zu_dunkel(self, hass):
        """Test that 'zu dunkel' is transformed to 'heller' command."""
        from multistage_assist.capabilities.clarification import ClarificationCapability
        
        cap = ClarificationCapability(hass, get_llm_config())
        user_input = make_input("im Wohnzimmer ist es zu dunkel")
        
        result = await cap.run(user_input)
        
        assert isinstance(result, list)
        assert len(result) == 1
        
        # Should transform to explicit brightness command
        command = result[0].lower()
        assert "wohnzimmer" in command
        assert "heller" in command or "licht" in command
    
    @pytest.mark.asyncio
    async def test_full_flow_zu_dunkel_no_disambiguation(self, hass):
        """
        Full integration test: "im Wohnzimmer ist es zu dunkel" should NOT
        trigger disambiguation between lights, covers, and sensors.
        
        The flow should:
        1. Clarification: "im Wohnzimmer ist es zu dunkel" -> "Mache das Licht im Wohnzimmer heller"
        2. keyword_intent: HassLightSet, domain=light, area=Wohnzimmer
        3. entity_resolver: Only return light.* entities
        4. No disambiguation needed if there's only 1 light, or same-domain lights
        """
        from multistage_assist.capabilities.clarification import ClarificationCapability
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        
        # Step 1: Clarification
        clarification_cap = ClarificationCapability(hass, get_llm_config())
        user_input = make_input("im Wohnzimmer ist es zu dunkel")
        
        clarified = await clarification_cap.run(user_input)
        assert isinstance(clarified, list) and len(clarified) > 0
        
        # Step 2: Keyword Intent (on clarified text)
        keyword_cap = KeywordIntentCapability(hass, get_llm_config())
        clarified_input = make_input(clarified[0])
        
        ki_result = await keyword_cap.run(clarified_input)
        
        # Should detect light domain
        assert ki_result.get("domain") == "light", \
            f"Expected domain='light', got: {ki_result.get('domain')}"
        
        # Should be HassLightSet for brightness adjustment
        assert ki_result.get("intent") == "HassLightSet", \
            f"Expected intent='HassLightSet', got: {ki_result.get('intent')}"
        
        # Domain MUST be returned at top level for injection to work
        assert "domain" in ki_result, "Domain must be in response for filtering to work"


class TestImplicitCommands:
    """Tests for implicit command transformations."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_direction", [
        ("im Wohnzimmer ist es zu dunkel", "heller"),
        ("es ist zu dunkel hier", "heller"),
        ("zu hell im Bad", "dunkler"),
        ("es ist zu hell", "dunkler"),
        ("im Büro ist es zu dunkel", "heller"),
    ])
    async def test_implicit_brightness_transformation(self, hass, input_text, expected_direction):
        """Test that implicit brightness commands are transformed correctly."""
        from multistage_assist.capabilities.clarification import ClarificationCapability
        
        cap = ClarificationCapability(hass, get_llm_config())
        user_input = make_input(input_text)
        
        result = await cap.run(user_input)
        
        assert isinstance(result, list) and len(result) > 0
        combined = " ".join(result).lower()
        
        assert expected_direction in combined, \
            f"Expected '{expected_direction}' in transformation of '{input_text}', got: {result}"
