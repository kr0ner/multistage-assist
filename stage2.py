import ast
import logging
from typing import Any, Dict, List, Optional, Callable

from homeassistant.components import conversation
from homeassistant.helpers import intent

from .prompt_executor import PromptExecutor
from .entity_resolver import EntityResolver
from .prompts import (
    PLURAL_SINGULAR_PROMPT,
    DISAMBIGUATION_PROMPT,
    DISAMBIGUATION_RESOLUTION_PROMPT,
    GET_VALUE_PHRASE_PROMPT,
    SENSOR_SELECTION_PROMPT,  # LLM returns {"entities": [...], "function": "lambda values: ..."} for get_value
)
from .stage0 import Stage0Result

_LOGGER = logging.getLogger(__name__)


def _with_new_text(
    user_input: conversation.ConversationInput, new_text: str
) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        language=user_input.language,
        agent_id=user_input.agent_id,
        device_id=user_input.device_id,
    )


class SafeLambdaError(Exception):
    pass


def _safe_compile_lambda(src: str) -> Callable[[List[float]], float]:
    """
    Safely parse and compile a lambda like 'lambda values: sum(values)/len(values)'
    Allowed nodes: Module, Expr, Lambda, arguments, arg, Name, Load, BinOp, Add, Sub, Mult, Div, FloorDiv,
                   Pow, Mod, UnaryOp, UAdd, USub, Call (sum, len, max, min, abs, round),
                   Constant, Tuple, List.
    Only parameter name 'values' is allowed.
    """
    if not src or "lambda" not in src:
        raise SafeLambdaError("Function must be a lambda.")

    try:
        tree = ast.parse(src, mode="eval")
    except Exception as e:
        raise SafeLambdaError(f"Parse error: {e}") from e

    allowed_calls = {"sum", "len", "max", "min", "abs", "round"}
    allowed_names = {"values"}

    class Validator(ast.NodeVisitor):
        def visit_Lambda(self, node: ast.Lambda):
            # Only single arg 'values'
            if not isinstance(node.args, ast.arguments) or len(node.args.args) != 1:
                raise SafeLambdaError("Lambda must have exactly one parameter 'values'.")
            if node.args.args[0].arg != "values":
                raise SafeLambdaError("Lambda parameter must be named 'values'.")
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if node.id not in allowed_names and node.id not in allowed_calls:
                raise SafeLambdaError(f"Name '{node.id}' not allowed.")
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                raise SafeLambdaError("Only builtins sum,len,max,min,abs,round are allowed.")
            self.generic_visit(node)

        def generic_visit(self, node):
            # Disallow comprehensions, attributes, subscripts beyond simple names, etc.
            forbidden = (
                ast.Attribute, ast.Dict, ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp,
                ast.Subscript, ast.IfExp, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Lambda  # (Lambda itself is ok but handled)
            )
            if isinstance(node, forbidden) and not isinstance(node, ast.Lambda):
                raise SafeLambdaError(f"Forbidden node: {type(node).__name__}")
            super().generic_visit(node)

    Validator().visit(tree)

    code = compile(tree, "<safe_lambda>", "eval")
    func = eval(code, {"__builtins__": {}}, {"sum": sum, "len": len, "max": max, "min": min, "abs": abs, "round": round})
    if not callable(func):
        raise SafeLambdaError("Compiled object is not callable.")
    return func  # type: ignore[return-value]


class Stage2Processor:
    """Stage 2: Execute intents, handle disambiguation, and compute get_value with LLM selection/aggregation."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self.entities = EntityResolver(hass)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _get_state_key(self, user_input) -> str:
        return getattr(user_input, "session_id", None) or user_input.conversation_id

    async def _is_plural(self, text: str) -> bool:
        data = await self.prompts.run(PLURAL_SINGULAR_PROMPT, {"user_input": text})
        is_plural = bool(data and data.get("multiple_entities"))
        _LOGGER.debug("Plural detection for '%s' -> %s (raw=%s)", text, is_plural, data)
        return is_plural

    async def _make_continuing_response(
        self, message: str, user_input: conversation.ConversationInput, end=False
    ) -> conversation.ConversationResult:
        resp = intent.IntentResponse(language=user_input.language or "de")
        resp.response_type = intent.IntentResponseType.QUERY_ANSWER
        resp.async_set_speech(message)
        return conversation.ConversationResult(
            response=resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=not end,
        )

    def has_pending(self, user_input) -> bool:
        return self._get_state_key(user_input) in self._pending

    async def resolve_pending(self, user_input):
        key = self._get_state_key(user_input)
        pending = self._pending.get(key)
        _LOGGER.debug("Resolving pending for %s: %s", key, pending)

        if not pending:
            # Nothing pending → delegate to default agent
            return await conversation.async_converse(
                self.hass,
                text=user_input.text,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language or "de",
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        # Resolution prompt
        data = await self.prompts.run(
            DISAMBIGUATION_RESOLUTION_PROMPT,
            {"user_input": user_input.text, "input_entities": pending["candidates"]},
            temperature=0.25,
        )
        _LOGGER.debug("Resolution output: %s", data)

        entities = (data or {}).get("entities") or []
        message = (data or {}).get("message") or ""
        action = (data or {}).get("action")

        if action == "abort":
            self._pending.pop(key, None)
            return await self._make_continuing_response("Okay, abgebrochen.", user_input, end=True)

        if not entities:
            return await self._make_continuing_response(
                "Entschuldigung, ich habe das nicht verstanden. Bitte wiederhole.", user_input
            )

        if pending["kind"] == "action":
            return await self._resolve_action(user_input, pending, entities, message)

        if pending["kind"] == "value":
            return await self._resolve_value(user_input, pending, entities)

        self._pending.pop(key, None)
        return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

    async def _resolve_action(self, user_input, pending, entities: List[str], message: str):
        intent_obj = pending.get("intent")
        self._pending.pop(self._get_state_key(user_input), None)
        if not intent_obj:
            return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

        try:
            intent_name = getattr(intent_obj, "name", None)
            last_resp = None
            for eid in entities:
                slots: Dict[str, Any] = {"name": {"value": eid}}
                _LOGGER.debug("Executing action intent '%s' with slots=%s", intent_name, slots)
                last_resp = await intent.async_handle(
                    self.hass,
                    platform="conversation",
                    intent_type=intent_name,
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context,
                    language=user_input.language or "de",
                )

            if last_resp and not last_resp.speech:
                if message:
                    last_resp.async_set_speech(message)
                else:
                    pretty = ", ".join(pending["candidates"].get(eid, eid) for eid in entities)
                    last_resp.async_set_speech(f"Okay, {pretty} ist erledigt.")

            return conversation.ConversationResult(
                response=last_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            )
        except Exception as e:
            _LOGGER.exception("Failed to execute action disambiguation: %s", e)
            return await self._make_continuing_response("Entschuldigung, das konnte ich nicht ausführen.", user_input)

    async def _resolve_value(self, user_input, pending, entities: List[str]):
        """Final pick for value read after disambiguation."""
        self._pending.pop(self._get_state_key(user_input), None)
        clarification_data = pending.get("clarification_data") or {}

        if len(entities) == 1:
            return await self._handle_get_value(user_input, entities[0], clarification_data)

        # multiple still → selection prompt again with function
        return await self._run_sensor_selection(user_input, entities, clarification_data)

    async def _handle_get_value(self, user_input, eid: str, data: Dict[str, Any]):
        """Read a single sensor value and phrase the answer."""
        state = self.hass.states.get(eid)
        if not state or state.state in ("unknown", "unavailable", None):
            return await self._make_continuing_response("Der Wert ist derzeit nicht verfügbar.", user_input)

        context = {
            "measurement": data.get("measurement") or "Wert",
            "value": state.state,
            "unit": state.attributes.get("unit_of_measurement") or "",
            "area": data.get("area"),
        }
        phrased = await self.prompts.run(GET_VALUE_PHRASE_PROMPT, context)
        message = (phrased or {}).get("message") or f"{context['measurement']}: {context['value']} {context['unit']}"
        return await self._make_continuing_response(message, user_input, end=True)

    async def _call_action_disambiguation(self, user_input, entity_ids: List[str], intent_obj=None):
        if await self._is_plural(user_input.text):
            # user said "alle ..." → send to default agent to operate on all
            return await conversation.async_converse(
                self.hass,
                text=user_input.text,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language or "de",
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        # Build mapping for prompt
        entity_map = await self.entities.make_entity_map(entity_ids)
        _LOGGER.debug("Action disambiguation candidates: %s", entity_map)

        data = await self.prompts.run(DISAMBIGUATION_PROMPT, {"input_entities": entity_map})
        msg = (data or {}).get("message") or "Bitte präzisiere, welches Gerät du meinst."

        key = self._get_state_key(user_input)
        self._pending[key] = {"kind": "action", "intent": intent_obj, "candidates": entity_map}
        return await self._make_continuing_response(msg, user_input)

    async def _call_value_disambiguation(self, user_input, entity_ids: List[str], clarification_data: Dict[str, Any]):
        entity_map = await self.entities.make_entity_map(entity_ids)
        _LOGGER.debug("Value disambiguation candidates: %s", entity_map)

        data = await self.prompts.run(DISAMBIGUATION_PROMPT, {"input_entities": entity_map})
        msg = (data or {}).get("message") or "Bitte präzisiere, welchen Sensor du meinst."

        key = self._get_state_key(user_input)
        self._pending[key] = {"kind": "value", "clarification_data": clarification_data, "candidates": entity_map}
        return await self._make_continuing_response(msg, user_input)

    async def _run_sensor_selection(self, user_input, sensors: List[str], clarification_data: Dict[str, Any]):
        """
        Ask LLM to pick sensors and return a Python lambda for aggregation.
        Expected response: {"entities": [...], "function": "lambda values: ..."}
        """
        selection = await self.prompts.run(
            SENSOR_SELECTION_PROMPT,
            {
                "user_input": user_input.text,
                "measurement": clarification_data.get("measurement"),
                # Do NOT pass the full candidates list to avoid context bloat per requirements.
            },
        )
        _LOGGER.debug("SENSOR_SELECTION_PROMPT output: %s", selection)

        chosen = [e for e in (selection or {}).get("entities", []) if isinstance(e, str)]
        func_src = (selection or {}).get("function")

        # If LLM named specific entities, intersect with known sensors to avoid hallucinations
        if chosen:
            chosen = [e for e in chosen if e in sensors]
        else:
            # If nothing selected explicitly, keep existing sensors
            chosen = list(sensors)

        # If still none, fail early
        if not chosen:
            return await self._make_continuing_response("Ich konnte keinen passenden Sensor finden.", user_input)

        if len(chosen) == 1 and not func_src:
            # Single sensor → just read it
            return await self._handle_get_value(user_input, chosen[0], clarification_data)

        # Multiple sensors or explicit function → aggregate
        # Collect numeric values
        values: List[float] = []
        units: List[str] = []
        for eid in chosen:
            st = self.hass.states.get(eid)
            if not st:
                continue
            try:
                val = float(str(st.state).replace(",", "."))
            except Exception:
                continue
            values.append(val)
            unit = st.attributes.get("unit_of_measurement")
            if unit:
                units.append(str(unit))

        if not values:
            return await self._make_continuing_response("Die Werte sind derzeit nicht verfügbar.", user_input)

        # Compile/execute safe lambda
        try:
            func = _safe_compile_lambda(func_src or "lambda values: sum(values)/len(values)")
            result_value = func(values)
        except Exception as e:
            _LOGGER.exception("Aggregation function failed: %s", e)
            # Fall back to average
            result_value = sum(values) / len(values)

        unit_out = units[0] if units else ""
        context = {
            "measurement": clarification_data.get("measurement") or "Wert",
            "value": result_value,
            "unit": unit_out,
            "area": clarification_data.get("area"),
        }
        phrased = await self.prompts.run(GET_VALUE_PHRASE_PROMPT, context)
        message = (phrased or {}).get("message") or f"{context['measurement']}: {result_value} {unit_out}"
        return await self._make_continuing_response(message, user_input, end=True)

    async def run(self, user_input: conversation.ConversationInput, s0: Stage0Result):
        """
        Execute intent resolved by Stage0.
        - For get_value: if many sensors, use LLM selection + math function.
        - For actions: if multiple entities, disambiguate (unless plural).
        """
        # If Stage0 already provided a narrowed set of ids, we use them.
        merged_ids = list(s0.resolved_ids or [])
        _LOGGER.debug("Stage2 received resolved ids: %s", merged_ids)

        # Special case: Get state / measurement queries
        intent_name = getattr(s0.intent, "name", None)
        if intent_name == "HassGetState":
            # Narrow to existing sensors only
            sensors = [eid for eid in merged_ids if eid.startswith("sensor.") and self.hass.states.get(eid) is not None]

            # Build faux "data" dict using Stage1-B schema if available (from DefaultAgent in prior turn)
            # We try to deduce measurement from user_input if needed via SENSOR_SELECTION_PROMPT step anyway.
            clarification_data = {
                "measurement": None,
                "area": None,
            }

            # If we have just one sensor → answer directly
            if len(sensors) == 1:
                return await self._handle_get_value(user_input, sensors[0], clarification_data)

            # If many sensors → ask LLM to select + compute
            if len(sensors) > 1:
                return await self._run_sensor_selection(user_input, sensors, clarification_data)

            # If there are no sensors, try disambiguation based on all ids we have
            if not sensors and merged_ids:
                return await self._call_value_disambiguation(user_input, merged_ids, clarification_data)

            return await self._make_continuing_response("Ich konnte keinen passenden Sensor finden.", user_input)

        # Control intents (turn on/off etc.)
        if len(merged_ids) == 0:
            return await self._make_continuing_response("Ich konnte kein passendes Gerät finden.", user_input)

        if len(merged_ids) == 1:
            # Execute immediately via DefaultAgent for reliability
            return await conversation.async_converse(
                self.hass,
                text=user_input.text,
                context=user_input.context,
                conversation_id=user_input.conversation_id,
                language=user_input.language or "de",
                agent_id=conversation.HOME_ASSISTANT_AGENT,
            )

        # Multiple candidates → disambiguate
        return await self._call_action_disambiguation(user_input, merged_ids, intent_obj=s0.intent)
