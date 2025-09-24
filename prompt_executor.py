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
    ) -> dict[str, Any]:
        """
        Run through escalation path until schema requirements are satisfied.
        The `prompt` must have keys: {"system": str, "schema": dict}.
        Always returns a dict ({} if nothing worked).
        """
        system_prompt = prompt["system"]
        schema = prompt.get("schema")

        # Add schema-based output format to the system prompt automatically
        if schema:
            system_prompt = system_prompt.strip() + self._schema_to_prompt(schema)

        for stage in self.escalation_path:
            result = await self._execute(stage, system_prompt, context, temperature)
            if result and self._validate_schema(result, schema):
                context.update(result)
                return result
            _LOGGER.info("Stage %s unable to handle, escalating...", stage.name)

        return {}

    @staticmethod
    def _schema_to_prompt(schema: dict) -> str:
        """Convert a JSON schema into an 'Output format' block for the LLM."""
        return "\n\n## Output format\n" + json.dumps(schema, indent=2, ensure_ascii=False)

    @staticmethod
    def _validate_schema(result: dict[str, Any], schema: dict | None) -> bool:
        if not schema:
            # fallback: require at least "message"
            return "message" in result
        for key in schema.get("required", []):
            if key not in result:
                return False
        return True

    async def _execute(
        self,
        stage: Stage,
        system_prompt: str,
        context: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any] | None:
        ip, port, model = _get_stage_config(self.config, stage)
        client = OllamaClient(ip, port)

        try:
            resp_text = await client.chat(
                model,
                system_prompt,
                json.dumps(context, ensure_ascii=False),
                temperature=temperature,
            )
            _LOGGER.debug("Stage %s raw response: %s", stage.name, resp_text)

            # Strip to JSON block if present
            if "{" in resp_text and "}" in resp_text:
                cleaned = resp_text[resp_text.find("{") : resp_text.rfind("}") + 1]
            else:
                cleaned = resp_text.strip()

            _LOGGER.debug("Stage %s cleaned response: %s", stage.name, cleaned)
            return json.loads(cleaned)
        except Exception as err:
            _LOGGER.warning("Stage %s execution failed: %s", stage.name, err)
            return None
