import logging
from typing import Any, Dict, List, Optional

from homeassistant.components import conversation
from homeassistant.components.conversation.default_agent import DefaultAgent
from hassil.recognize import recognize_best

from .entity_resolver import EntityResolver
from .prompt_executor import PromptExecutor
from .prompts import ENTITY_FILTER_PROMPT

_LOGGER = logging.getLogger(__name__)


class Stage0Result:
    """Container for Stage0 output passed to Stage2."""
    def __init__(self, type_: str, intent=None, raw=None, resolved_ids: Optional[List[str]] = None):
        self.type = type_
        self.intent = intent
        self.raw = raw
        self.resolved_ids = resolved_ids or []


class Stage0Processor:
    """Stage 0: Dry-run NLU and early entity resolution with LLM-based filtering when needed."""

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.prompts = PromptExecutor(config)
        self.entities = EntityResolver(hass)

    async def _dry_run_recognize(self, user_input: conversation.ConversationInput):
        agent = conversation.async_get_agent(self.hass)
        if not isinstance(agent, DefaultAgent):
            _LOGGER.warning("Only works with DefaultAgent right now")
            return None

        language = user_input.language or "de"
        lang_intents = await agent.async_get_or_load_intents(language)
        if lang_intents is None:
            _LOGGER.debug("No intents loaded for language=%s", language)
            return None

        slot_lists = await agent._make_slot_lists()
        intent_context = agent._make_intent_context(user_input)

        def _run():
            return recognize_best(
                user_input.text,
                lang_intents.intents,
                slot_lists=slot_lists,
                intent_context=intent_context,
                language=language,
                best_metadata_key="hass_custom_sentence",
                best_slot_name="name",
            )

        _LOGGER.debug("Running dry-run recognize for utterance='%s'", user_input.text)
        return await self.hass.async_add_executor_job(_run)

    def _apply_filter_hints(self, candidates: List[str], hints: Dict[str, Any]) -> List[str]:
        """Apply simple attribute-based filter hints returned by LLM (without sending candidate list)."""
        if not isinstance(hints, dict):
            return candidates

        name_f = (hints.get("name") or "").lower()
        dc_f = (hints.get("device_class") or "").lower()
        unit_f = (hints.get("unit") or "").lower()
        area_f = (hints.get("area") or "").lower()
        domain_f = (hints.get("domain") or "").lower()
        must_include: List[str] = [s.lower() for s in hints.get("must_include", []) if isinstance(s, str)]
        must_exclude: List[str] = [s.lower() for s in hints.get("must_exclude", []) if isinstance(s, str)]

        filtered: List[str] = []
        for eid in candidates:
            state = self.hass.states.get(eid)
            if not state:
                continue
            attrs = state.attributes or {}
            fname = (attrs.get("friendly_name") or "").lower()
            dev_class = (attrs.get("device_class") or "").lower()
            unit = (attrs.get("unit_of_measurement") or "").lower()
            area = (attrs.get("area_id") or attrs.get("area") or "").lower()
            domain = eid.split(".", 1)[0].lower()

            # Positive filters
            if name_f and name_f not in fname and name_f not in eid.lower():
                continue
            if dc_f and dc_f != dev_class:
                continue
            if unit_f and unit_f not in unit:
                continue
            if area_f and area_f not in area and area_f not in fname and area_f not in eid.lower():
                continue
            if domain_f and domain_f != domain:
                continue

            # Must include words anywhere in id or friendly name
            if must_include and not all(any(token in s for s in (eid.lower(), fname)) for token in must_include):
                continue

            # Exclusions
            if any(token in eid.lower() or token in fname for token in must_exclude):
                continue

            filtered.append(eid)
        return filtered

    async def run(self, user_input: conversation.ConversationInput) -> Stage0Result | None:
        """Return None (no intent), Stage0Result('clarification'), or Stage0Result('intent')."""
        result = await self._dry_run_recognize(user_input)
        if not result or not result.intent:
            _LOGGER.debug("NLU did not produce an intent.")
            return None

        # Resolve entities from NLU slots
        entities = {k: v.value for k, v in (result.entities or {}).items()}
        _LOGGER.debug("NLU extracted entities: %s", entities)
        resolved = await self.entities.resolve(entities)
        _LOGGER.debug(
            "Resolved entity_ids: by_area=%s, by_name=%s, merged=%s",
            resolved.by_area, resolved.by_name, resolved.merged
        )

        if not resolved.merged:
            _LOGGER.debug("No entities resolved → Stage1 clarification")
            return Stage0Result("clarification", raw=result)

        # Early LLM-based filtering when list is large.
        threshold = int(getattr(self.config, "early_filter_threshold", 10))
        merged_ids = list(resolved.merged)
        if len(merged_ids) > threshold:
            _LOGGER.debug("Too many entities (%d) → ask Stage1 filter hints", len(merged_ids))
            hints = await self.prompts.run(ENTITY_FILTER_PROMPT, {"user_input": user_input.text})
            _LOGGER.debug("ENTITY_FILTER_PROMPT hints: %s", hints)
            merged_ids = self._apply_filter_hints(merged_ids, hints)
            _LOGGER.debug("After filter hints: %d candidates", len(merged_ids))

        if not merged_ids:
            _LOGGER.debug("All candidates filtered away → Stage1 clarification")
            return Stage0Result("clarification", raw=result)

        return Stage0Result("intent", intent=result.intent, raw=result, resolved_ids=merged_ids)
