import json
import enum
import logging
from typing import Any

try:
    from .ollama_client import OllamaClient
    from .const import (
        CONF_STAGE1_IP,
        CONF_STAGE1_PORT,
        CONF_STAGE1_MODEL,
    )
    from .utils.json_utils import extract_json_from_llm_string
except (ImportError, ValueError):
    from ollama_client import OllamaClient
    from const import (
        CONF_STAGE1_IP,
        CONF_STAGE1_PORT,
        CONF_STAGE1_MODEL,
    )
    try:
        from utils.json_utils import extract_json_from_llm_string
    except ImportError:
        from multistage_assist.utils.json_utils import extract_json_from_llm_string

_LOGGER = logging.getLogger(__name__)


class Stage(enum.Enum):
    STAGE1 = 1


DEFAULT_ESCALATION_PATH: list[Stage] = [Stage.STAGE1]


def _get_stage_config(config: dict, stage: Stage) -> tuple[str, int, str]:
    if stage == Stage.STAGE1:
        return (
            config[CONF_STAGE1_IP],
            config[CONF_STAGE1_PORT],
            config[CONF_STAGE1_MODEL],
        )
    # Stage 2 logic is now handled by GoogleGeminiClient, not here.
    raise ValueError(f"Unknown stage: {stage}")


class PromptExecutor:
    """Runs LLM prompts with automatic escalation and shared context."""

    def __init__(self, config: dict, escalation_path: list[Stage] | None = None):
        self.config = config
        self.escalation_path = escalation_path or DEFAULT_ESCALATION_PATH

    async def run(
        self,
        prompt: dict[str, Any],
        context: dict[str, Any],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any] | list | None:
        """
        Run through escalation path until schema requirements are satisfied.
        The `prompt` must have keys: {"system": str, "schema": dict}.
        Always returns {} or [] if nothing worked.
        """
        system_prompt = prompt["system"]
        schema = prompt.get("schema")

        for stage in self.escalation_path:
            result = await self._execute(stage, system_prompt, context, temperature, schema)
            if result is None:
                _LOGGER.info("Stage %s returned None, escalating...", stage.name)
                continue

            if self._validate_schema(result, schema):
                if isinstance(result, dict):
                    context.update(result)
                return result

            _LOGGER.info(
                "Stage %s produced output but did not satisfy schema. Got=%s",
                stage.name,
                result,
            )

        return [] if (schema and schema.get("type") == "array") else {}

    # Removed old _schema_to_prompt which injected string descriptions of the JSON schema

    @staticmethod
    def _validate_schema(result: Any, schema: dict | None) -> bool:
        if not schema:
            return bool(result)

        def _is_type(val, t) -> bool:
            if t == "string":
                return isinstance(val, str)
            if t == "boolean":
                return isinstance(val, bool)
            if t == "object":
                return isinstance(val, dict)
            if t == "array":
                return isinstance(val, list)
            if t == "null":
                return val is None
            return True

        stype = schema.get("type")

        # Array schema
        if stype == "array":
            if not isinstance(result, list):
                return False
            item_type = schema.get("items", {}).get("type")
            if not item_type:
                return True
            return all(_is_type(x, item_type) for x in result)

        # Object schema (or any schema with "properties")
        if stype == "object" or "properties" in schema:
            if not isinstance(result, dict):
                return False

            props = schema.get("properties", {}) or {}
            required = schema.get("required", [])
            for key, spec in props.items():
                if key not in result:
                    if key in required:
                        return False
                    continue

                expected = spec.get("type")
                val = result[key]

                # Union types like ["string", "null"] or ["array", "null"]
                if isinstance(expected, list):
                    if not any(_is_type(val, t) for t in expected):
                        return False
                elif expected == "array":
                    if not isinstance(val, list):
                        return False
                    item_t = spec.get("items", {}).get("type")
                    if item_t:
                        if not all(_is_type(x, item_t) for x in val):
                            return False
                else:
                    if val is None and expected != "null":
                        return False
                    if val is not None and not _is_type(val, expected or "string"):
                        return False
            return True

        return bool(result)

    async def _execute(
        self,
        stage: Stage,
        system_prompt: str,
        context: dict[str, Any],
        temperature: float,
        schema: dict | None = None,
    ) -> dict[str, Any] | list | None:
        ip, port, model = _get_stage_config(self.config, stage)
        client = OllamaClient(ip, port)
        try:
            resp_text = await client.chat(
                model,
                system_prompt,
                json.dumps(context, ensure_ascii=False),
                temperature=temperature,
                format=schema,  # Pass native JSON schema for Structured Outputs
            )
            
            _LOGGER.debug("Stage %s raw response: %s", stage.name, resp_text)
            print(f"\nRAW OLLAMA RESPONSE: {repr(resp_text)}")
            print(f"RAW OLLAMA SCHEMA PASSED: {json.dumps(schema)}")
            
            # Since we're using Native Structured Outputs, Ollama guarantees valid JSON.
            # However, some models still wrap the output in markdown code blocks like ```json ... ```
            return extract_json_from_llm_string(resp_text)

        except Exception as err:
            _LOGGER.warning("Stage %s execution failed (%s): %s", stage.name, type(err).__name__, err)
            return None
