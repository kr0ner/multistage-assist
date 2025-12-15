import logging
import re
from typing import Any, Dict, List, Optional
from homeassistant.components import conversation
from .base_stage import BaseStage

# Capabilities
from .capabilities.clarification import ClarificationCapability
from .capabilities.disambiguation import DisambiguationCapability
from .capabilities.disambiguation_select import DisambiguationSelectCapability
from .capabilities.plural_detection import PluralDetectionCapability
from .capabilities.intent_confirmation import IntentConfirmationCapability
from .capabilities.yes_no_response import YesNoResponseCapability
from .capabilities.intent_executor import IntentExecutorCapability
from .capabilities.entity_resolver import EntityResolverCapability
from .capabilities.keyword_intent import KeywordIntentCapability
from .capabilities.area_alias import AreaAliasCapability
from .capabilities.memory import MemoryCapability
from .capabilities.intent_resolution import IntentResolutionCapability
from .capabilities.timer import TimerCapability
from .capabilities.command_processor import CommandProcessorCapability
from .capabilities.vacuum import VacuumCapability
from .capabilities.calendar import CalendarCapability
from .capabilities.semantic_cache import SemanticCacheCapability

from .conversation_utils import (
    make_response,
    error_response,
    with_new_text,
    filter_candidates_by_state,
)
from .stage_result import Stage0Result

_LOGGER = logging.getLogger(__name__)


class Stage1Processor(BaseStage):
    """Stage 1: Handles clarification, disambiguation, and multi-command orchestration."""

    name = "stage1"
    capabilities = [
        ClarificationCapability,
        DisambiguationCapability,
        DisambiguationSelectCapability,
        PluralDetectionCapability,
        IntentConfirmationCapability,
        IntentExecutorCapability,
        EntityResolverCapability,
        KeywordIntentCapability,
        AreaAliasCapability,
        MemoryCapability,
        IntentResolutionCapability,
        TimerCapability,
        CommandProcessorCapability,
        VacuumCapability,
        YesNoResponseCapability,
        CalendarCapability,
        SemanticCacheCapability,  # Semantic command cache
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._pending: Dict[str, Dict[str, Any]] = {}

        # Inject shared memory into capabilities that need it
        memory = self.get("memory")

        if self.has("entity_resolver"):
            self.get("entity_resolver").set_memory(memory)

        if self.has("intent_resolution"):
            self.get("intent_resolution").set_memory(memory)
        
        # Inject semantic cache into command processor for verified command storage
        if self.has("semantic_cache") and self.has("command_processor"):
            self.get("command_processor").set_cache(self.get("semantic_cache"))

    async def _normalize_area_aliases(self, user_input):
        """
        Preprocess user input to normalize common area aliases using memory.
        E.g., "bad" → "Badezimmer", "ezi" → "Esszimmer"

        This reduces unnecessary disambiguation by using correct area names early.
        """
        text = user_input.text
        words = text.lower().split()

        # Check each word for area aliases
        for word in words:
            # Strip common punctuation
            clean_word = word.strip(".,!?")
            if not clean_word:
                continue

            # Look up in memory
            memory_cap = self.get("memory")
            normalized = await memory_cap.get_area_alias(clean_word)
            if normalized:
                # Case-insensitive replacement
                _LOGGER.debug(
                    "[Stage1] Early normalization: '%s' → '%s'", clean_word, normalized
                )
                # Replace in text (preserve case for first letter if capitalized)
                import re

                pattern = re.compile(re.escape(clean_word), re.IGNORECASE)
                text = pattern.sub(normalized, text, count=1)

        # Return modified input if changed
        if text != user_input.text:
            _LOGGER.debug("[Stage1] Normalized text: %s → %s", user_input.text, text)
            return with_new_text(user_input, text)
        return user_input

    async def run(self, user_input, prev_result=None):
        _LOGGER.debug("[Stage1] Input='%s'", user_input.text)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        # 1. Handle Pending
        if key in self._pending:
            _LOGGER.debug("[Stage1] Found pending state for key=%s", key)
            return await self._handle_pending(key, user_input)

        # 2. Handle Stage0
        if (
            isinstance(prev_result, Stage0Result)
            and len(prev_result.resolved_ids or []) > 1
        ):
            _LOGGER.debug(
                "[Stage1] Refining Stage0 result with %d candidates",
                len(prev_result.resolved_ids),
            )
            return await self._handle_stage0_result(prev_result, user_input)

        # 3. Handle New Command
        return await self._handle_new_command(user_input, prev_result)

    # --- Handlers ---

    async def _handle_pending(self, key: str, user_input) -> Dict[str, Any]:
        pending = self._pending.pop(key, None)
        if not pending:
            return {"status": "error", "result": await error_response(user_input)}

        ptype = pending.get("type")

        # Timer Follow-up
        if ptype == "timer":
            timer = self.get("timer")
            res = await timer.continue_flow(user_input, pending)
            return self._handle_processor_result(key, res)

        # Calendar Follow-up
        if ptype == "calendar":
            calendar = self.get("calendar")
            res = await calendar.continue_flow(user_input, pending)
            return self._handle_processor_result(key, res)

        # Learning Confirmation
        if ptype == "learning_confirmation":
            _LOGGER.debug(
                "[Stage1] Processing learning confirmation: '%s'", user_input.text
            )
            if user_input.text.lower().strip() in (
                "ja",
                "ja bitte",
                "gerne",
                "okay",
                "mach das",
            ):
                memory = self.get("memory")
                src, tgt = pending["source"], pending["target"]
                l_type = pending.get("learning_type")

                if l_type == "entity":
                    await memory.learn_entity_alias(src, tgt)
                elif l_type == "floor":
                    await memory.learn_floor_alias(src, tgt)
                else:
                    await memory.learn_area_alias(src, tgt)

                return {
                    "status": "handled",
                    "result": await make_response("Alles klar, gemerkt.", user_input),
                }
            return {
                "status": "handled",
                "result": await make_response("Okay, nicht gemerkt.", user_input),
            }

        # Disambiguation
        if ptype == "disambiguation":
            processor = self.get("command_processor")
            res = await processor.continue_disambiguation(user_input, pending)

            if res.get("status") == "handled" and pending.get("remaining"):
                return await self._continue_sequence(
                    user_input, res["result"], pending["remaining"]
                )

            return self._handle_processor_result(key, res)

        # Area Clarification (entity not found, user provided area)
        if ptype == "area_clarification":
            area_text = user_input.text.strip()
            intent_name = pending.get("intent")
            domain = pending.get("domain")
            original_name = pending.get("name")
            slots = pending.get("slots", {})
            
            _LOGGER.debug(
                "[Stage1] Processing area clarification: area='%s', intent='%s', domain='%s'",
                area_text, intent_name, domain
            )
            
            # Resolve the area using area_alias
            area_alias_cap = self.get("area_alias")
            areas = [a.name for a in self.hass.helpers.area_registry.async_get_registry().areas.values()]
            alias_result = await area_alias_cap.run(
                user_input, user_query=area_text, candidates=areas
            )
            
            matched_area = alias_result.get("match") if alias_result else None
            if not matched_area or matched_area == "GLOBAL":
                matched_area = area_text  # Use raw input as fallback
            
            _LOGGER.debug("[Stage1] Area resolved to: '%s'", matched_area)
            
            # Now get all entities in that area for the domain
            resolver = self.get("entity_resolver")
            entity_ids = await resolver.run(
                user_input,
                domain=domain,
                area=matched_area,
            )
            
            if not entity_ids:
                msg = f"Ich konnte keine {domain}-Geräte in '{matched_area}' finden."
                return {
                    "status": "handled",
                    "result": await make_response(msg, user_input),
                }
            
            # Update slots with the area
            slots["area"] = matched_area
            
            # Process the command with resolved entities
            processor = self.get("command_processor")
            res = await processor.process(
                user_input,
                entity_ids,
                intent_name,
                {k: v for k, v in slots.items() if k not in ("name", "entity_id")},
                # Offer to learn the alias if original name was unusual
                {"type": "entity", "source": original_name, "target": entity_ids[0]} if len(entity_ids) == 1 else None,
            )
            return self._handle_processor_result(key, res)

        return {"status": "error", "result": await error_response(user_input)}

    async def _handle_new_command(self, user_input, prev_result) -> Dict[str, Any]:
        """
        Process a completely new command through the full capability chain.
        """
        # --- Semantic Cache Check (Fast Path) ---
        if self.has("semantic_cache"):
            cache = self.get("semantic_cache")
            cached = await cache.lookup(user_input.text)
            if cached and cached["score"] > 0.9:
                _LOGGER.info(
                    "[Stage1] Cache HIT (%.3f): %s -> %s",
                    cached["score"], cached["intent"], cached["entity_ids"]
                )
                
                if cached["required_disambiguation"]:
                    # User had to choose before - need to prompt again
                    _LOGGER.debug("[Stage1] Cached command required disambiguation, prompting user")
                    
                    # Use disambiguation capability with cached options
                    options = cached.get("disambiguation_options") or {}
                    if options:
                        disamb_cap = self.get("disambiguation")
                        disamb_result = await disamb_cap.run(
                            user_input,
                            intent=cached["intent"],
                            candidates=list(options.keys()),
                            params=cached["slots"],
                        )
                        if disamb_result:
                            return disamb_result
                else:
                    # No disambiguation needed - execute directly
                    processor = self.get("command_processor")
                    res = await processor.process(
                        user_input,
                        cached["entity_ids"],
                        cached["intent"],
                        {k: v for k, v in cached["slots"].items() if k not in ("name", "entity_id")},
                        None,  # No learning data from cache
                    )
                    return self._handle_processor_result(
                        getattr(user_input, "session_id", None) or user_input.conversation_id,
                        res,
                    )
        
        # --- Early Area Normalization (Optimization) ---
        # Normalize common area aliases BEFORE clarification to reduce disambiguation
        # E.g., "bad" → "Badezimmer" before LLM processing
        normalized_input = await self._normalize_area_aliases(user_input)

        # --- Capability Chain ---
        # 1. Clarification (split multi-commands, rephrase)
        clarified = await self.use("clarification", normalized_input)

        if not clarified or (isinstance(clarified, list) and not clarified):
            return {"status": "escalate", "result": prev_result}

        if isinstance(clarified, list):
            norm = (user_input.text or "").strip().lower()
            atomic = [c for c in clarified if isinstance(c, str) and c.strip()]

            # Single command optimization: if clarification returns a single command
            # that's very similar to input, skip the sequence recursion
            if len(atomic) == 1:
                clarified_norm = atomic[0].strip().lower()
                
                # Exact match or very similar (allows minor normalization)
                is_unchanged = (
                    clarified_norm == norm
                    or norm in clarified_norm  # Input is contained in output
                    or clarified_norm.replace(".", "").strip() == norm.replace(".", "").strip()
                )
                
                if is_unchanged:
                    ki_data = await self.use("keyword_intent", user_input) or {}
                    intent_name = ki_data.get("intent")
                    slots = ki_data.get("slots") or {}

                    # --- FALLBACK: No domain detected, try entity name matching ---
                    if not intent_name:
                        _LOGGER.debug(
                            "[Stage1] No domain detected, trying entity name fallback for: '%s'",
                            user_input.text
                        )
                        fallback_result = await self._try_entity_name_fallback(user_input)
                        if fallback_result:
                            return fallback_result

                    # --- TIMER INTERCEPT ---
                    if intent_name == "HassTimerSet":
                        target_name = slots.get("name")
                        if target_name:
                            memory = self.get("memory")
                            known_id = await memory.get_entity_alias(target_name)
                            if known_id:
                                _LOGGER.debug(
                                    "[Stage1] Memory hit for timer device: %s -> %s",
                                    target_name,
                                    known_id,
                                )
                                slots["device_id"] = known_id

                        res = await self.get("timer").run(user_input, intent_name, slots)
                        return self._handle_processor_result(
                            getattr(user_input, "session_id", None)
                            or user_input.conversation_id,
                            res,
                        )
                    # -----------------------

                    # --- VACUUM INTERCEPT ---
                    if intent_name == "HassVacuumStart":
                        return await self.get("vacuum").run(user_input, intent_name, slots)
                    # ------------------------

                    # --- CALENDAR INTERCEPT ---
                    if intent_name in ("HassCalendarCreate", "HassCreateEvent", "HassCalendarAdd"):
                        res = await self.get("calendar").run(user_input, intent_name, slots)
                        return self._handle_processor_result(
                            getattr(user_input, "session_id", None)
                            or user_input.conversation_id,
                            res,
                        )
                    # --------------------------

                    # --- TEMPORARY CONTROL INTERCEPT ---
                    # HassTemporaryControl needs resolution then passes to command_processor
                    # which handles the timebox script execution
                    if intent_name == "HassTemporaryControl":
                        _LOGGER.debug(
                            "[Stage1] HassTemporaryControl detected with duration=%s",
                            slots.get("duration"),
                        )
                        # Resolve entities first (pass ki_data to avoid duplicate LLM call)
                        res_data = await self.get("intent_resolution").run(user_input, ki_data=ki_data)
                        if not res_data:
                            return {"status": "escalate", "result": prev_result}

                        processor = self.get("command_processor")
                        res = await processor.process(
                            user_input,
                            res_data["entity_ids"],
                            intent_name,  # Pass HassTemporaryControl
                            {
                                k: v
                                for k, v in res_data["slots"].items()
                                if k not in ("name", "entity_id")
                            },
                            res_data.get("learning_data"),
                        )
                        return self._handle_processor_result(
                            getattr(user_input, "session_id", None)
                            or user_input.conversation_id,
                            res,
                        )
                    # ------------------------------------

                    res_data = await self.get("intent_resolution").run(user_input, ki_data=ki_data)
                    if not res_data:
                        # If we have a name but no entity found, ask for area
                        # Store pending state so follow-up is handled properly
                        name_slot = slots.get("name")
                        if name_slot and intent_name in ("HassTurnOn", "HassTurnOff", "HassLightSet", "HassSetPosition"):
                            _LOGGER.debug(
                                "[Stage1] Entity not found for name '%s', asking for area", name_slot
                            )
                            msg = f"Ich konnte '{name_slot}' nicht finden. In welchem Bereich ist das?"
                            key = getattr(user_input, "session_id", None) or user_input.conversation_id
                            self._pending[key] = {
                                "type": "area_clarification",
                                "intent": intent_name,
                                "name": name_slot,
                                "domain": domain,
                                "slots": slots,
                            }
                            return {
                                "status": "handled",
                                "result": await make_response(msg, user_input),
                                "pending_data": self._pending[key],
                            }
                        return {"status": "escalate", "result": prev_result}

                    processor = self.get("command_processor")
                    res = await processor.process(
                        user_input,
                        res_data["entity_ids"],
                        res_data["intent"],
                        {
                            k: v
                            for k, v in res_data["slots"].items()
                            if k not in ("name", "entity_id")
                        },
                        res_data.get("learning_data"),
                    )
                    return self._handle_processor_result(
                        getattr(user_input, "session_id", None)
                        or user_input.conversation_id,
                        res,
                    )

            # Clarification changed the input or multiple commands - use sequence executor
            if len(atomic) > 0:
                return await self._execute_sequence(user_input, atomic)

        return {"status": "escalate", "result": prev_result}

    def _handle_processor_result(self, key, res: Dict[str, Any]) -> Dict[str, Any]:
        if res.get("pending_data"):
            _LOGGER.debug(
                "[Stage1] Storing pending data for key=%s: %s",
                key,
                res["pending_data"]["type"],
            )
            self._pending[key] = res["pending_data"]

        # Log final speech for debugging
        if res.get("result") and res["result"].response.speech:
            speech = res["result"].response.speech.get("plain", {}).get("speech", "")
            _LOGGER.debug("[Stage1] Final Response Speech: '%s'", speech)

        return res

    # ... (Keep _execute_sequence, _continue_sequence, _add_confirmation_if_needed, _merge_speech) ...
    async def _execute_sequence(
        self, user_input, commands: List[str], previous_results: List[Any] = None
    ) -> Dict[str, Any]:
        return await self._continue_sequence(user_input, None, commands)

    async def _continue_sequence(
        self, user_input, prev_res, commands: List[str]
    ) -> Dict[str, Any]:
        results = [prev_res] if prev_res else []
        agent = getattr(self, "agent", None)
        key = getattr(user_input, "session_id", None) or user_input.conversation_id

        for i, cmd in enumerate(commands):
            _LOGGER.debug("[Stage1] Sequence %d/%d: %s", i + 1, len(commands), cmd)
            res = await agent._run_pipeline(with_new_text(user_input, cmd))

            if key in self._pending:
                remaining = commands[i + 1 :]
                self._pending[key]["remaining"] = remaining
                if results:
                    self._merge_speech(res, results)
                return {"status": "handled", "result": res}

            results.append(res)

        final = results[-1]
        self._merge_speech(final, results[:-1])
        return {"status": "handled", "result": final}

    async def _add_confirmation_if_needed(
        self, user_input, result, intent_name, entity_ids, params=None
    ):
        if not result or not result.response:
            return
        speech = (
            result.response.speech.get("plain", {}).get("speech", "")
            if result.response.speech
            else ""
        )
        if not speech or speech.strip() == "Okay.":
            confirm_cap = self.get("intent_confirmation")
            gen_data = await confirm_cap.run(
                user_input,
                intent_name=intent_name,
                entity_ids=entity_ids,
                params=params,
            )
            msg = gen_data.get("message")
            if msg:
                result.response.async_set_speech(msg)

    def _merge_speech(
        self,
        final: conversation.ConversationResult,
        results: List[conversation.ConversationResult],
    ):
        texts = []
        for r in results:
            speech = r.response.speech.get("plain", {}).get("speech", "")
            texts.append(speech)

        final_speech = final.response.speech.get("plain", {}).get("speech", "")
        texts.append(final_speech)

        full_text = " ".join(texts)
        final.response.async_set_speech(full_text)

    async def _handle_stage0_result(
        self, prev_result: Stage0Result, user_input
    ) -> Dict[str, Any]:
        res_data = await self.use("intent_resolution", user_input)

        candidates = list(prev_result.resolved_ids)
        intent_name = (prev_result.intent or "").strip()
        params = {}
        learning_data = None

        if res_data:
            candidates = res_data["entity_ids"]
            intent_name = res_data["intent"]
            params = {
                k: v
                for k, v in res_data["slots"].items()
                if k not in ("name", "entity_id")
            }
            learning_data = res_data.get("learning_data")

        processor = self.get("command_processor")
        res = await processor.process(
            user_input, candidates, intent_name, params, learning_data
        )
        return self._handle_processor_result(
            getattr(user_input, "session_id", None) or user_input.conversation_id, res
        )

    async def _try_entity_name_fallback(self, user_input) -> Optional[Dict[str, Any]]:
        """
        Fallback when no domain is detected: try to find entities whose name
        matches the input text using fuzzy matching.
        
        This helps with commands like "Türklingel aus" where "Türklingel" might
        match an automation entity name but isn't in our keyword list.
        """
        from .utils.fuzzy_utils import fuzzy_match
        from homeassistant.helpers import entity_registry as er
        
        text = user_input.text.lower().strip()
        
        # Determine the command intent from common words
        text_lower = f" {text} "
        if any(word in text_lower for word in [" an ", " ein ", " auf ", " aktivier", " start", " öffne"]):
            intent = "HassTurnOn"
            command = "on"
        elif any(word in text_lower for word in [" aus ", " ab ", " deaktivier", " stopp", " schließe"]):
            intent = "HassTurnOff"
            command = "off"
        else:
            # Can't determine intent
            return None
        
        # Check for duration (HassTemporaryControl)
        duration = None
        duration_match = re.search(r'für\s+(\d+)\s*(minuten?|sekunden?|stunden?)', text, re.IGNORECASE)
        if duration_match:
            duration = duration_match.group(0)
            intent = "HassTemporaryControl"
        
        # Get all entities and try fuzzy matching
        ent_reg = er.async_get(self.hass)
        candidates = []
        
        for entity in ent_reg.entities.values():
            if entity.disabled_by:
                continue
            
            # Check friendly name
            state = self.hass.states.get(entity.entity_id)
            if state:
                friendly_name = state.attributes.get("friendly_name", "")
                if friendly_name:
                    candidates.append({
                        "id": entity.entity_id,
                        "name": friendly_name.lower(),
                        "domain": entity.domain
                    })
            
            # Also check entity_id object_id part
            obj_id = entity.entity_id.split(".", 1)[-1].replace("_", " ")
            candidates.append({
                "id": entity.entity_id,
                "name": obj_id.lower(),
                "domain": entity.domain
            })
        
        # Extract potential entity name from text (remove common command words)
        name_part = text
        for word in ["schalte", "mache", "aktiviere", "deaktiviere", "stelle", 
                     "an", "aus", "ein", "ab", "auf", "zu", "die", "den", "das", "der",
                     "bitte", "mal", "für", "minuten", "minute", "sekunden", "sekunde", "stunden", "stunde"]:
            name_part = re.sub(rf'\b{re.escape(word)}\b', '', name_part, flags=re.IGNORECASE)
        
        # Remove duration numbers
        name_part = re.sub(r'\d+', '', name_part)
        name_part = ' '.join(name_part.split()).strip()
        
        if not name_part or len(name_part) < 3:
            return None
        
        _LOGGER.debug("[Stage1] Fallback: searching for entity matching '%s'", name_part)
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = fuzzy_match(name_part, candidate["name"])
            if score > best_score and score >= 80:
                best_score = score
                best_match = candidate
        
        if not best_match:
            _LOGGER.debug("[Stage1] Fallback: no entity match found for '%s'", name_part)
            return None
        
        _LOGGER.info(
            "[Stage1] Fallback matched entity: '%s' -> %s (score: %d)",
            name_part, best_match["id"], best_score
        )
        
        # Build params
        params = {"command": command}
        if duration:
            params["duration"] = duration
        
        # Execute via command processor
        processor = self.get("command_processor")
        res = await processor.process(
            user_input,
            [best_match["id"]],
            intent,
            params,
            {"alias": name_part, "entity_id": best_match["id"]},  # learning data
        )
        
        return self._handle_processor_result(
            getattr(user_input, "session_id", None) or user_input.conversation_id,
            res,
        )
