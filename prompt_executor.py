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
except (ImportError, ValueError):
    from ollama_client import OllamaClient
    from const import (
        CONF_STAGE1_IP,
        CONF_STAGE1_PORT,
        CONF_STAGE1_MODEL,
    )

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

        if schema:
            system_prompt = system_prompt.strip() + self._schema_to_prompt(schema)

        for stage in self.escalation_path:
            result = await self._execute(stage, system_prompt, context, temperature)
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

    @staticmethod
    def _schema_to_prompt(schema: dict) -> str:
        """
        Generate a STRICT output-format block that nudges small models
        to return valid, minified JSON and nothing else.
        """
        if not schema:
            return ""

        # Array schema
        if schema.get("type") == "array":
            item_type = (schema.get("items") or {}).get("type", "string")
            return (
                "\n\n## Output format (STRICT)\n"
                f"Return ONLY a minified JSON array of {item_type}s.\n"
                "- No text, no markdown, no backticks.\n"
                '- Example: ["item1","item2"]\n'
                '- Invalid: ```["item1"]```  or  Here you go: ["item1"]'
            )

        # Object schema (default)
        props = (schema or {}).get("properties") or {}
        if not props:
            return (
                "\n\n## Output format (STRICT)\n"
                "Return ONLY a minified JSON object.\n"
                "- No text, no markdown, no backticks.\n"
                "- Example: {}"
            )

        keys = list(props.keys())

        def _slot(t: str, spec: dict) -> str:
            if t == "array":
                it = (spec.get("items") or {}).get("type", "string")
                return f"[<{it}>]"
            return f"<{t}>"

        example_pairs = []
        for k, spec in props.items():
            t = spec.get("type", "string")
            example_pairs.append(f'"{k}":{_slot(t, spec)}')

        example_obj = "{" + ",".join(example_pairs) + "}"

        lines = [
            "\n\n## Output format (STRICT)",
            "Return ONLY a **minified JSON object** with **exactly** these keys and **no others**, in this order:",
            ", ".join(f'"{k}"' for k in keys),
            "",
            "- No text, no explanations, no markdown, no backticks.",
            f"- Example shape: {example_obj}",
            '- Invalid: {"unexpected":true}, ```{...}```, or any leading/trailing text.',
        ]

        return "\n".join(lines)

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
            for key, spec in props.items():
                if key not in result:
                    return False

                expected = spec.get("type")
                val = result[key]

                # Union types like ["string", "null"] or ["array", "null"]
                if isinstance(expected, list):
                    return any(_is_type(val, t) for t in expected)

                if expected == "array":
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
            # tolerant JSON block extraction
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
