"""Tests for BaseStage (REQ-QUAL-001, REQ-TEST-002, REQ-PIPE-001).

Covers: BaseStage init, capability injection, has/get/use pattern.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from multistage_assist.base_stage import BaseStage
from multistage_assist.capabilities.base import Capability


class DummyCapability(Capability):
    name = "dummy"
    description = "test"

    async def run(self, user_input, **kwargs):
        return {"ran": True, **kwargs}


class DummyCapabilityWithMemory(Capability):
    name = "needs_memory"
    description = "test with memory"

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.memory = None

    def set_memory(self, mem):
        self.memory = mem

    async def run(self, user_input, **kwargs):
        return {"memory": self.memory is not None}


class MemoryCapability(Capability):
    name = "memory"
    description = "memory"

    async def run(self, user_input, **kwargs):
        return {}


class ConcreteStage(BaseStage):
    name = "test_stage"
    capabilities = [DummyCapability]

    async def process(self, user_input, context=None):
        pass


class StageWithMemoryInjection(BaseStage):
    name = "test_stage_memory"
    capabilities = [MemoryCapability, DummyCapabilityWithMemory]

    async def process(self, user_input, context=None):
        pass


class TestBaseStage:
    """Tests for BaseStage initialization and capability management."""

    def test_capabilities_map_populated(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        assert "dummy" in stage.capabilities_map

    def test_has_returns_true_for_existing(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        assert stage.has("dummy") is True

    def test_has_returns_false_for_missing(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        assert stage.has("nonexistent") is False

    def test_get_returns_capability(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        cap = stage.get("dummy")
        assert isinstance(cap, DummyCapability)

    def test_get_raises_for_missing(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        with pytest.raises(KeyError, match="nonexistent"):
            stage.get("nonexistent")

    @pytest.mark.asyncio
    async def test_use_calls_capability_run(self):
        hass = MagicMock()
        stage = ConcreteStage(hass, {})
        result = await stage.use("dummy", MagicMock(), extra="value")
        assert result["ran"] is True
        assert result["extra"] == "value"


class TestBaseStageInjection:
    """Tests for automatic dependency injection."""

    def test_memory_auto_injected(self):
        """Verify that set_memory is called on capabilities that have it."""
        hass = MagicMock()
        stage = StageWithMemoryInjection(hass, {})
        cap = stage.get("needs_memory")
        assert cap.memory is not None

    @pytest.mark.asyncio
    async def test_injected_memory_accessible_in_run(self):
        hass = MagicMock()
        stage = StageWithMemoryInjection(hass, {})
        result = await stage.use("needs_memory", MagicMock())
        assert result["memory"] is True
