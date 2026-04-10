"""Tests for prompt_executor.py — PromptExecutor, _get_stage_config, _validate_schema."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from multistage_assist.prompt_executor import (
    PromptExecutor,
    Stage,
    DEFAULT_ESCALATION_PATH,
    _get_stage_config,
)
from multistage_assist.const import CONF_STAGE1_IP, CONF_STAGE1_PORT, CONF_STAGE1_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides):
    base = {CONF_STAGE1_IP: "127.0.0.1", CONF_STAGE1_PORT: 11434, CONF_STAGE1_MODEL: "qwen3:4b"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _get_stage_config
# ---------------------------------------------------------------------------

class TestGetStageConfig:
    def test_stage1_returns_ip_port_model(self):
        cfg = _config()
        ip, port, model = _get_stage_config(cfg, Stage.STAGE1)
        assert (ip, port, model) == ("127.0.0.1", 11434, "qwen3:4b")

    def test_unknown_stage_raises(self):
        cfg = _config()
        fake_stage = MagicMock(name="FAKE")
        fake_stage.name = "FAKE"
        # Only STAGE1 is valid; anything else should raise
        with pytest.raises(ValueError, match="Unknown stage"):
            _get_stage_config(cfg, fake_stage)


# ---------------------------------------------------------------------------
# PromptExecutor.__init__
# ---------------------------------------------------------------------------

class TestPromptExecutorInit:
    def test_default_escalation_path(self):
        pe = PromptExecutor(_config())
        assert pe.escalation_path == DEFAULT_ESCALATION_PATH

    def test_custom_escalation_path(self):
        path = [Stage.STAGE1, Stage.STAGE1]
        pe = PromptExecutor(_config(), escalation_path=path)
        assert pe.escalation_path is path


# ---------------------------------------------------------------------------
# _validate_schema (static, no I/O)
# ---------------------------------------------------------------------------

class TestValidateSchema:
    """REQ-QUAL-006 — unit-level coverage for schema validation logic."""

    def test_no_schema_truthy(self):
        assert PromptExecutor._validate_schema({"a": 1}, None) is True

    def test_no_schema_falsy(self):
        assert PromptExecutor._validate_schema({}, None) is False
        assert PromptExecutor._validate_schema([], None) is False

    # --- array schemas ---
    def test_array_valid(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert PromptExecutor._validate_schema(["a", "b"], schema) is True

    def test_array_wrong_type(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert PromptExecutor._validate_schema("not a list", schema) is False

    def test_array_item_type_mismatch(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert PromptExecutor._validate_schema([1, 2], schema) is False

    def test_array_no_item_type(self):
        schema = {"type": "array"}
        assert PromptExecutor._validate_schema([1, "x", None], schema) is True

    # --- object schemas ---
    def test_object_valid(self):
        schema = {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        }
        assert PromptExecutor._validate_schema({"intent": "HassTurnOn"}, schema) is True

    def test_object_missing_required(self):
        schema = {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        }
        assert PromptExecutor._validate_schema({}, schema) is False

    def test_object_optional_missing_ok(self):
        schema = {
            "type": "object",
            "properties": {"intent": {"type": "string"}, "area": {"type": "string"}},
            "required": ["intent"],
        }
        assert PromptExecutor._validate_schema({"intent": "HassTurnOn"}, schema) is True

    def test_object_wrong_value_type(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "boolean"}},
            "required": ["count"],
        }
        assert PromptExecutor._validate_schema({"count": "nope"}, schema) is False

    def test_object_null_value_not_allowed(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert PromptExecutor._validate_schema({"name": None}, schema) is False

    def test_object_null_allowed_in_union(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
            "required": ["name"],
        }
        assert PromptExecutor._validate_schema({"name": None}, schema) is True
        assert PromptExecutor._validate_schema({"name": "Alice"}, schema) is True

    def test_object_nested_array_property(self):
        schema = {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["ids"],
        }
        assert PromptExecutor._validate_schema({"ids": ["a", "b"]}, schema) is True
        assert PromptExecutor._validate_schema({"ids": [1]}, schema) is False
        assert PromptExecutor._validate_schema({"ids": "not_a_list"}, schema) is False

    def test_properties_without_explicit_type(self):
        """Schema with 'properties' but no explicit 'type' field still validates as object."""
        schema = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
        assert PromptExecutor._validate_schema({"x": "hi"}, schema) is True
        assert PromptExecutor._validate_schema("string", schema) is False


# ---------------------------------------------------------------------------
# run() — async orchestration
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_successful_first_stage(self):
        pe = PromptExecutor(_config())
        with patch.object(pe, "_execute", new_callable=AsyncMock, return_value={"intent": "HassTurnOn"}):
            prompt = {"system": "you are helpful", "schema": {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}}
            result = await pe.run(prompt, {})
        assert result == {"intent": "HassTurnOn"}

    @pytest.mark.asyncio
    async def test_escalation_on_none(self):
        """When first execute returns None, second is tried."""
        pe = PromptExecutor(_config(), escalation_path=[Stage.STAGE1, Stage.STAGE1])
        calls = [None, {"intent": "HassTurnOff"}]
        with patch.object(pe, "_execute", new_callable=AsyncMock, side_effect=calls):
            prompt = {"system": "sys", "schema": {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}}
            result = await pe.run(prompt, {})
        assert result == {"intent": "HassTurnOff"}

    @pytest.mark.asyncio
    async def test_escalation_on_schema_failure(self):
        """Result that doesn't match schema triggers escalation."""
        pe = PromptExecutor(_config(), escalation_path=[Stage.STAGE1, Stage.STAGE1])
        calls = [{"wrong_key": 1}, {"intent": "ok"}]
        with patch.object(pe, "_execute", new_callable=AsyncMock, side_effect=calls):
            prompt = {"system": "sys", "schema": {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}}
            result = await pe.run(prompt, {})
        assert result == {"intent": "ok"}

    @pytest.mark.asyncio
    async def test_all_stages_fail_returns_empty_dict(self):
        pe = PromptExecutor(_config())
        with patch.object(pe, "_execute", new_callable=AsyncMock, return_value=None):
            prompt = {"system": "sys", "schema": {"type": "object"}}
            result = await pe.run(prompt, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_all_stages_fail_returns_empty_list_for_array_schema(self):
        pe = PromptExecutor(_config())
        with patch.object(pe, "_execute", new_callable=AsyncMock, return_value=None):
            prompt = {"system": "sys", "schema": {"type": "array"}}
            result = await pe.run(prompt, {})
        assert result == []

    @pytest.mark.asyncio
    async def test_context_updated_on_dict_result(self):
        pe = PromptExecutor(_config())
        with patch.object(pe, "_execute", new_callable=AsyncMock, return_value={"intent": "On"}):
            prompt = {"system": "sys", "schema": {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}}
            ctx = {"existing": True}
            await pe.run(prompt, ctx)
        assert ctx["intent"] == "On"
        assert ctx["existing"] is True


# ---------------------------------------------------------------------------
# _execute() — OllamaClient integration boundary
# ---------------------------------------------------------------------------

class TestExecute:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        pe = PromptExecutor(_config())
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value='{"intent": "HassTurnOn"}')
        with patch("multistage_assist.prompt_executor.OllamaClient", return_value=mock_client):
            result = await pe._execute(Stage.STAGE1, "system", {"text": "hi"}, 0.0, None)
        assert result == {"intent": "HassTurnOn"}
        mock_client.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        pe = PromptExecutor(_config())
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=ConnectionError("offline"))
        with patch("multistage_assist.prompt_executor.OllamaClient", return_value=mock_client):
            result = await pe._execute(Stage.STAGE1, "system", {}, 0.0, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_schema_passed_to_client(self):
        pe = PromptExecutor(_config())
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value='{"x": "val"}')
        with patch("multistage_assist.prompt_executor.OllamaClient", return_value=mock_client):
            await pe._execute(Stage.STAGE1, "sys", {}, 0.0, schema)
        _, kwargs = mock_client.chat.call_args
        assert kwargs.get("format") == schema or mock_client.chat.call_args[0][4] == schema
