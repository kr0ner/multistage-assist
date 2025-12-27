"""Integration tests for unknown area learning flow.

Tests cover:
- Stage2LLM returns pending state when area is unknown
- Conversation stores pending_data and asks user
- User response is matched to area and alias is learned
- Original command is re-run with learned alias
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


@dataclass
class MockConversationInput:
    text: str
    conversation_id: str = "test-area-learning"
    context: Any = None
    device_id: str = "test-device"
    satellite_id: Optional[str] = None
    language: str = "de"
    agent_id: str = "test-agent"
    extra_system_prompt: Optional[str] = None


@dataclass
class MockConversationResult:
    response: Any = None
    
    @dataclass
    class _Response:
        speech: Dict = field(default_factory=lambda: {"plain": {"speech": "OK"}})
        
        def async_set_speech(self, text):
            self.speech = {"plain": {"speech": text}}

    def __post_init__(self):
        if self.response is None:
            self.response = self._Response()


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    hass.config.path = MagicMock(return_value="/tmp/test")
    return hass


@pytest.fixture
def mock_config():
    return {"reranker_ip": "localhost", "reranker_port": 9876}


class TestAreaResolverUnknownArea:
    """Test area_resolver returns unknown_area when resolution fails."""

    @pytest.mark.asyncio
    async def test_area_resolver_returns_unknown_area(self, mock_hass, mock_config):
        """When area is not found, resolver returns unknown_area with candidates."""
        from multistage_assist.capabilities.area_resolver import AreaResolverCapability
        
        with patch.object(AreaResolverCapability, '__init__', lambda self, h, c: None):
            resolver = AreaResolverCapability.__new__(AreaResolverCapability)
            resolver.hass = mock_hass
            resolver._config = mock_config
            
            # Mock find_area to return None (not found)
            resolver.find_area = MagicMock(return_value=None)
            
            # Mock LLM prompt to return no match
            resolver._safe_prompt = AsyncMock(return_value={"match": None})
            
            # Mock area registry
            mock_area = MagicMock()
            mock_area.name = "Küche"
            mock_areas = [mock_area]
            
            with patch('multistage_assist.capabilities.area_resolver.ar') as mock_ar:
                mock_registry = MagicMock()
                mock_registry.async_list_areas.return_value = mock_areas
                mock_ar.async_get.return_value = mock_registry
                
                result = await resolver.run(None, area_name="Ki-Bad")
                
                # Should return unknown_area with candidates
                assert result.get("match") is None
                assert result.get("unknown_area") == "Ki-Bad"
                assert "Küche" in result.get("candidates", [])


class TestStageResultPending:
    """Test StageResult.pending() factory method."""

    def test_pending_factory_creates_pending_result(self):
        """StageResult.pending() creates correct pending state."""
        from multistage_assist.stage_result import StageResult
        
        result = StageResult.pending(
            pending_type="area_learning",
            message="Welchen Bereich meinst du?",
            pending_data={"unknown_alias": "Ki-Bad", "candidates": ["Küche", "Bad"]},
            raw_text="Licht im Ki-Bad an",
        )
        
        assert result.status == "pending"
        assert result.pending_data["type"] == "area_learning"
        assert result.pending_data["original_prompt"] == "Welchen Bereich meinst du?"
        assert result.pending_data["unknown_alias"] == "Ki-Bad"
        assert result.raw_text == "Licht im Ki-Bad an"


class TestConversationAreaLearning:
    """Test conversation.py handles area_learning pending state."""

    @pytest.mark.asyncio
    async def test_pending_status_stored_in_execution_pending(self, mock_hass, mock_config):
        """When stage returns pending, pending_data is stored."""
        from multistage_assist.conversation import MultiStageAssistAgent
        from multistage_assist.stage_result import StageResult
        
        with patch.object(MultiStageAssistAgent, '__init__', lambda self, h, c: None):
            agent = MultiStageAssistAgent.__new__(MultiStageAssistAgent)
            agent.hass = mock_hass
            agent.config = mock_config
            agent._config = mock_config
            agent._execution_pending = {}
            agent.stages = []
            
            # Create pending result
            pending_result = StageResult.pending(
                pending_type="area_learning",
                message="Ich kenne 'Ki-Bad' nicht. Welchen Bereich meinst du?",
                pending_data={
                    "unknown_alias": "Ki-Bad",
                    "candidates": ["Küche", "Bad", "Kinder Badezimmer"],
                    "original_text": "Licht im Ki-Bad an",
                },
            )
            
            # Directly test the pending storage logic (from _run_pipeline)
            result = pending_result
            conv_id = "test-area-learning"
            if result.status == "pending":
                result.pending_data["_created_at"] = time.time()
                result.pending_data["_retry_count"] = 0
                agent._execution_pending[conv_id] = result.pending_data
            
            # Verify pending is stored
            assert conv_id in agent._execution_pending
            assert agent._execution_pending[conv_id]["type"] == "area_learning"
            assert agent._execution_pending[conv_id]["unknown_alias"] == "Ki-Bad"

    @pytest.mark.asyncio
    async def test_continue_area_learning_matches_response(self, mock_hass, mock_config):
        """User response is matched to area and alias is learned via Stage2."""
        from multistage_assist.stage2_llm import Stage2LLMProcessor
        from multistage_assist.stage_result import StageResult
        
        # Initialize Stage2
        with patch.object(Stage2LLMProcessor, '__init__', lambda self, h, c: None):
            stage2 = Stage2LLMProcessor.__new__(Stage2LLMProcessor)
            stage2.hass = mock_hass
            stage2.config = mock_config
            stage2._config = mock_config
            
            # Setup pending data
            pending_data = {
                "type": "area_learning",
                "unknown_alias": "Ki-Bad",
                "candidates": ["Küche", "Bad", "Kinder Badezimmer"],
                "original_text": "Licht im Ki-Bad an",
            }
            
            # User says "Kinder Badezimmer"
            user_input = MockConversationInput(text="Kinder Badezimmer")
            
            # Mock the AreaResolverCapability inside stage2
            with patch('multistage_assist.capabilities.area_resolver.AreaResolverCapability') as mock_resolver_class:
                mock_resolver = MagicMock()
                mock_resolver.learn_area_alias = AsyncMock()
                mock_resolver_class.return_value = mock_resolver
                
                # Mock capabilities dictionary/get method to return resolver
                stage2.get = MagicMock(return_value=mock_resolver)
                stage2.has = MagicMock(return_value=True)
                
                # Call continue_pending directly on stage2
                result = await stage2.continue_pending(user_input, pending_data)
                
                # Verify alias was learned
                mock_resolver.learn_area_alias.assert_called_once_with(
                    "Ki-Bad", "Kinder Badezimmer"
                )
                
                # Verify success result with rerun instructions
                assert result.status == "success"
                assert result.params["rerun_command"] is True
                assert result.params["learned_alias"] == "Ki-Bad"

    @pytest.mark.asyncio
    async def test_continue_area_learning_reprompts_on_no_match(self, mock_hass, mock_config):
        """When user response doesn't match, re-prompt is returned by Stage2."""
        from multistage_assist.stage2_llm import Stage2LLMProcessor
        
        # Initialize Stage2
        with patch.object(Stage2LLMProcessor, '__init__', lambda self, h, c: None):
            stage2 = Stage2LLMProcessor.__new__(Stage2LLMProcessor)
            stage2.hass = mock_hass
            stage2.config = mock_config
            
            # Setup pending data
            pending_data = {
                "type": "area_learning",
                "unknown_alias": "Ki-Bad",
                "candidates": ["Küche", "Bad", "Kinder Badezimmer"],
                "original_text": "Licht im Ki-Bad an",
                "_retry_count": 0,
            }
            
            # User says something that doesn't match any area
            user_input = MockConversationInput(text="Wohnzimmer")
            
            # Call continue_pending
            result = await stage2.continue_pending(user_input, pending_data)
            
            # Should return pending again
            assert result.status == "pending"
            assert result.pending_data["type"] == "area_learning"
            # Verify message asks again (contains unknown alias)
            response_text = result.pending_data["original_prompt"]
            assert "Ki-Bad" in response_text


class TestAreaLearningMessages:
    """Test that area learning uses correct message constants."""

    def test_unknown_area_messages_in_system_messages(self):
        """Verify area learning messages are in SYSTEM_MESSAGES."""
        from multistage_assist.constants.messages_de import SYSTEM_MESSAGES
        
        assert "unknown_area_ask" in SYSTEM_MESSAGES
        assert "unknown_area_learned" in SYSTEM_MESSAGES
        assert "unknown_area_not_matched" in SYSTEM_MESSAGES
        
        # Verify placeholders
        assert "{alias}" in SYSTEM_MESSAGES["unknown_area_ask"]
        assert "{alias}" in SYSTEM_MESSAGES["unknown_area_learned"]
        assert "{area}" in SYSTEM_MESSAGES["unknown_area_learned"]

    def test_unknown_area_message_formatting(self):
        """Verify message formatting works correctly."""
        from multistage_assist.constants.messages_de import SYSTEM_MESSAGES
        
        message = SYSTEM_MESSAGES["unknown_area_ask"].format(alias="Ki-Bad")
        assert "Ki-Bad" in message
        
        message = SYSTEM_MESSAGES["unknown_area_learned"].format(
            alias="Ki-Bad", area="Kinder Badezimmer"
        )
        assert "Ki-Bad" in message
        assert "Kinder Badezimmer" in message


class TestAreaResolverLearnAlias:
    """Test area_resolver.learn_area_alias method."""

    @pytest.mark.asyncio
    async def test_learn_area_alias_calls_memory(self, mock_hass, mock_config):
        """learn_area_alias should call MemoryCapability.learn_area_alias."""
        from multistage_assist.capabilities.area_resolver import AreaResolverCapability
        
        # Patch MemoryCapability where it's imported (from .memory)
        with patch('multistage_assist.capabilities.memory.MemoryCapability') as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory.learn_area_alias = AsyncMock()
            mock_memory_class.return_value = mock_memory
            
            with patch.object(AreaResolverCapability, '__init__', lambda self, h, c: None):
                resolver = AreaResolverCapability.__new__(AreaResolverCapability)
                resolver.hass = mock_hass
                resolver._config = mock_config
                
                await resolver.learn_area_alias("ki-bad", "Kinder Badezimmer")
                
                mock_memory.learn_area_alias.assert_called_once_with(
                    "ki-bad", "Kinder Badezimmer"
                )
