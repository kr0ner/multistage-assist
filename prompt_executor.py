import json
import enum
import logging
from typing import Any
from .ollama_client import OllamaClient
from .const import (
    CONF_STAGE1_IP,
    CONF_STAGE1_PORT,
    CONF_STAGE1_MODEL,
    CONF_STAGE2_IP,
    CONF_STAGE2_PORT,
    CONF_STAGE2_MODEL,
)

_LOGGER = logging.getLogger(__name__)


class Stage(enum.Enum):
    """Enum for Multi-Stage Assist stages."""
    STAGE1 = 1
    STAGE2 = 2
    # STAGE3 = 3 (future)


DEFAULT_ESCALATION_PATH: list[Stage] = [Stage.STAGE1, Stage.STAGE2]


def _get_stage_config(config: dict, stage: Stage) -> tuple[str, int, str]:
    if stage == Stage.STAGE1:
        return (
            config[CONF_STAGE1_IP],
            config[CONF_STAGE1_PORT],
            config[CONF_STAGE1_MODEL],
        )
    if stage == Stage.STAGE2:
        return (
            config[CONF_STAGE2_IP],
            config[CONF_STAGE2_PORT],
            config[CONF_STAGE2_MODEL],
        )
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

        # Add schema-based output format to the system prompt automatically
        if schema:
            system_prompt = system_prompt.strip() + self._schema_to_prompt(schema)

        for stage in self.escalation_path:
            result = await self._execute(stage, system_prompt, context, temperature)

            if result is None:
                _LOGGER.info("Stage %s returned None (execution/parsing error), escalating...", stage.name)
                continue

            if self._validate_schema(result, schema):
                _LOGGER.debug("Stage %s validated successfully against schema: %s", stage.name, schema)
                if isinstance(result, dict):
                    context.update(result)
                return result

            _LOGGER.info(
                "Stage %s produced output but did not satisfy schema. Got=%s",
                stage.name,
                result,
            )

        _LOGGER.warning("All stages exhausted without valid result for prompt.")
        if schema and schema.get("type") == "array":
            return []
        return {}

    @staticmethod
    def _schema_to_prompt(schema: dict) -> str:
        """Emit a clear 'Output format' block with field names and types, no examples."""
        if not schema:
            return ""

        schema_type = schema.get("type")

        if schema_type == "array":
            item_type = schema.get("items", {}).get("type", "string")
            return (
                "\n\n## Output format\n"
                f"Return ONLY a JSON array of {item_type}s."
            )

        props = (schema or {}).get("properties", {}) or {}
        if not props:
            return ""

        lines = ["\n\n## Output format", "Return ONLY a JSON object with the following fields:"]
        for key, spec in props.items():
            typ = spec.get("type", "string")
            if typ == "array":
                item_type = spec.get("items", {}).get("type", "string")
                lines.append(f'- "{key}": array of {item_type}s')
            else:
                lines.append(f'- "{key}": {typ}')
        return "\n".join(lines)

    @staticmethod
    def _validate_schema(result: Any, schema: dict | None) -> bool:
        if not schema:
            return bool(result)

        schema_type = schema.get("type")

        # --- Array schema ---
        if schema_type == "array":
            if not isinstance(result, list):
                return False
            item_spec = schema.get("items", {})
            item_type = item_spec.get("type")
            if item_type:
                for item in result:
                    if item_type == "string" and not isinstance(item, str):
                        return False
                    if item_type == "object" and not isinstance(item, dict):
                        return False
                    if item_type == "boolean" and not isinstance(item, bool):
                        return False
            return True

        # --- Object schema ---
        if schema_type == "object" or "properties" in schema:
            if not isinstance(result, dict):
                return False

            props = schema.get("properties", {}) or {}
            if not props:
                return True

            # All declared properties are implicitly required
            for key, spec in props.items():
                if key not in result:
                    return False
                if not isinstance(spec, dict):
                    continue
                typ = spec.get("type")

                # Handle union types like ["string", "null"]
                if isinstance(typ, list):
                    if result[key] is None and "null" in typ:
                        continue
                    if "string" in typ and isinstance(result[key], str):
                        continue
                    if "boolean" in typ and isinstance(result[key], bool):
                        continue
                    if "array" in typ and isinstance(result[key], list):
                        item_spec = spec.get("items", {})
                        item_type = item_spec.get("type")
                        if item_type == "string" and not all(isinstance(x, str) for x in result[key]):
                            return False
                        continue
                    return False

                # Simple types
                if typ == "boolean" and not isinstance(result[key], bool):
                    return False
                if typ == "string" and not (isinstance(result[key], str) or result[key] is None):
                    return False
                if typ == "array":
                    if not isinstance(result[key], list):
                        return False
                    item_spec = spec.get("items", {})
                    item_type = item_spec.get("type")
                    if item_type == "string" and not all(isinstance(x, str) for x in result[key]):
                        return False
            return True

        return bool(result)

    async def _execute(
        self,
        stage: Stage,
        system_prompt: str,
        context: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any] | list | None:
        ip, port, model = _get_stage_config(self.config, stage)
        client = OllamaClient(ip, port)

        try:
            resp_text = await client.chat(
                model,
                system_prompt,
                json.dumps(context, ensure_ascii=False),
                temperature=temperature,
            )

            # Try to extract JSON block
            if "[" in resp_text and "]" in resp_text:
                cleaned = resp_text[resp_text.find("[") : resp_text.rfind("]") + 1]
            elif "{" in resp_text and "}" in resp_text:
                cleaned = resp_text[resp_text.find("{") : resp_text.rfind("}") + 1]
            else:
                cleaned = resp_text.strip()

            _LOGGER.debug("Stage %s cleaned response: %s", stage.name, cleaned)
            return json.loads(cleaned)
        except Exception as err:
            _LOGGER.warning("Stage %s execution failed: %s", stage.name, err)
            return None
