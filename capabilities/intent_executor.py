import asyncio
import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import intent as ha_intent
from homeassistant.core import Context
from homeassistant.components.conversation import ConversationResult

from ..conversation_utils import (
    join_names,
    parse_duration_string,
    format_seconds_to_string,
)
from ..utils.response_builder import build_confirmation
from ..constants.entity_keywords import FRACTION_VALUES, DOMAIN_NAMES_PLURAL
from ..constants.messages_de import (
    ERROR_MESSAGES, 
    SYSTEM_MESSAGES,
    COMMAND_STATE_MAP,
    OPPOSITE_STATE_MAP,
    DURATION_TEMPLATES,
    DEFAULT_DEVICE_WORD,
    ACTION_VERBS,
    CONFIRMATION_TEMPLATES,
    get_state_response,
    get_opposite_state_word
)
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentExecutorCapability(Capability):
    """Execute a known HA intent for one or more concrete entity_ids."""

    name = "intent_executor"
    description = "Execute concrete Home Assistant intents for specific entities. Features: 1. Automatic Knowledge Graph prerequisite resolution 2. Parameter normalization (German fractions to integers) 3. Relative step adjustments (brightness/cover) 4. Advanced timebox/delay support via specialized scripts 5. State-query filtering and 6. Post-execution verification."

    RESOLUTION_KEYS = {"area", "floor", "name", "entity_id"}
    BRIGHTNESS_STEP = 35  # Percentage of current brightness for step_up/step_down
    COVER_STEP = 25       # Percentage for cover step_up/step_down (0=closed, 100=open)
    TIMEBOX_SCRIPT_ENTITY_ID = "script.timebox_entity_state"
    DELAY_SCRIPT_ENTITY_ID = "script.delay_action"

    def _extract_duration(self, params: Dict[str, Any]) -> tuple[int, int]:
        """Extract minutes and seconds from params. Returns (minutes, seconds)."""
        duration_raw = params.get("duration")
        if duration_raw:
            seconds = parse_duration_string(duration_raw)
            return (seconds // 60, seconds % 60)
        return (0, 0)

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters using German utils (fractions to integers)."""
        if not params:
            return {}
        
        normalized = params.copy()
        # key candidates for normalization
        for key in ["position", "brightness", "percentage"]:
            if key in normalized:
                val = normalized[key]
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    if val_lower in FRACTION_VALUES:
                        normalized[key] = FRACTION_VALUES[val_lower]
                        _LOGGER.debug("[IntentExecutor] Normalized fraction '%s': '%s' -> %d", key, val, normalized[key])
                    else:
                        # Handle "25 %", "25 Prozent", "25%"
                        import re
                        match = re.search(r"(\d+)\s*(?:%|prozent)", val_lower)
                        if match:
                            normalized[key] = int(match.group(1))
                            _LOGGER.debug("[IntentExecutor] Normalized percentage string '%s': '%s' -> %d", key, val, normalized[key])
                        else:
                            # Try simple int conversion if it's just a string number
                            try:
                                normalized[key] = int(val_lower.replace("%", "").strip())
                            except ValueError:
                                pass
        return normalized

    def _check_script_exists(self, script_entity_id: str) -> bool:
        """Check if a script entity exists in Home Assistant."""
        state = self.hass.states.get(script_entity_id)
        return state is not None

    async def _call_timebox_script(
        self,
        entity_id: str,
        minutes: int,
        seconds: int,
        value: int = None,
        action: str = None,
    ) -> bool:
        """Call timebox_entity_state script with value or action.
        
        Returns True on success, False on failure.
        """
        # Check if script is installed
        if not self._check_script_exists(self.TIMEBOX_SCRIPT_ENTITY_ID):
            _LOGGER.error(
                "[IntentExecutor] Script '%s' not found! "
                "Please install it from: multistage_assist/scripts/timebox_entity_state.yaml",
                self.TIMEBOX_SCRIPT_ENTITY_ID
            )
            return False
        
        _LOGGER.debug(
            "[IntentExecutor] Calling timebox script for %s: value=%s, action=%s, duration=%dm%ds",
            entity_id,
            value,
            action,
            minutes,
            seconds,
        )
        data = {"target_entity": entity_id, "minutes": minutes, "seconds": seconds}
        if value is not None:
            data["value"] = value
        if action is not None:
            data["action"] = action

        try:
            # Fire-and-forget - don't wait for the script to complete
            # (script waits for duration before reverting)
            await self.hass.services.async_call(
                "script", "timebox_entity_state", data, blocking=False
            )
            return True
        except Exception as e:
            _LOGGER.error(
                "[IntentExecutor] Timebox script failed for %s: %s", entity_id, e
            )
            return False

    async def _call_delay_script(
        self,
        entity_id: str,
        minutes: int,
        seconds: int,
        value: int = None,
        action: str = None,
    ) -> bool:
        """Call delay_action script to delay an action.
        
        Returns True on success, False on failure.
        """
        # Check if script is installed
        if not self._check_script_exists(self.DELAY_SCRIPT_ENTITY_ID):
            _LOGGER.error(
                "[IntentExecutor] Script '%s' not found! "
                "Please install it from: multistage_assist/scripts/delay_action.yaml",
                self.DELAY_SCRIPT_ENTITY_ID
            )
            return False
        
        _LOGGER.debug(
            "[IntentExecutor] Calling delay script for %s: value=%s, action=%s, delay=%dm%ds",
            entity_id,
            value,
            action,
            minutes,
            seconds,
        )
        data = {"target_entity": entity_id, "minutes": minutes, "seconds": seconds}
        if value is not None:
            data["value"] = value
        if action is not None:
            data["action"] = action

        try:
            await self.hass.services.async_call(
                "script", "delay_action", data, blocking=False
            )
            return True
        except Exception as e:
            _LOGGER.error(
                "[IntentExecutor] Delay script failed for %s: %s", entity_id, e
            )
            return False

    def _parse_delay_or_time(self, delay_str: str) -> tuple[int, int]:
        """Parse delay string ('10 Minuten') or time string ('15:30', '15 Uhr').
        
        Returns (minutes, seconds) for delay.
        For time strings, calculates delay from now to target time.
        If target time is in the past, schedules for next day.
        """
        import re
        from datetime import datetime, timedelta
        
        if not delay_str:
            return (0, 0)
        
        delay_str = delay_str.strip().lower()
        
        # Try parsing as time (HH:MM or HH Uhr)
        time_match = re.match(r'^(\d{1,2}):(\d{2})(?:\s*uhr)?$', delay_str)
        if not time_match:
            time_match = re.match(r'^(\d{1,2})\s*uhr$', delay_str)
            if time_match:
                hour = int(time_match.group(1))
                minute = 0
            else:
                # Not a time, try parsing as duration
                total_seconds = parse_duration_string(delay_str)
                return (total_seconds // 60, total_seconds % 60)
        else:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.lastindex >= 2 else 0
        
        # Calculate delay from now to target time
        import homeassistant.util.dt as dt_util
        now = dt_util.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If target time is in the past, schedule for tomorrow
        if target <= now:
            target += timedelta(days=1)
        
        delta = target - now
        total_seconds = int(delta.total_seconds())
        
        _LOGGER.debug(
            "[IntentExecutor] Parsed time '%s' -> target %s, delay=%d seconds",
            delay_str, target.strftime("%H:%M"), total_seconds
        )
        
        return (total_seconds // 60, total_seconds % 60)
    
    # Intent to action mapping for Knowledge Graph
    INTENT_TO_ACTION = {
        "HassTurnOn": "turn_on",
        "HassTurnOff": "turn_off",
        "HassLightSet": "turn_on",
        "HassSetPosition": "set_position",
        "HassClimateSetTemperature": "set_temperature",
        "HassVacuumStart": "turn_on",
        "HassMediaPause": "media_pause",
        "HassMediaResume": "media_play",
        "HassMediaPlay": "media_play",
        "HassMediaStop": "media_stop",
        "HassTimerSet": "start",
        "HassCoverOpen": "open",
        "HassCoverClose": "close",
    }
    
    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.knowledge_graph = None

    def set_knowledge_graph(self, kg_cap):
        """Inject knowledge graph capability."""
        self.knowledge_graph = kg_cap

    async def _resolve_prerequisites(
        self, entity_ids: List[str], intent_name: str
    ) -> List[Dict[str, Any]]:
        """Resolve and execute Knowledge Graph prerequisites for entities.
        
        This handles power dependencies and device coupling with AUTO mode.
        """
        if not self.knowledge_graph:
            return []
        
        action = self.INTENT_TO_ACTION.get(intent_name, "turn_on")
        executed_prerequisites = []
        
        for entity_id in entity_ids:
            resolution = await self.knowledge_graph.resolve_for_action(entity_id, action)
            
            # Execute AUTO prerequisites
            for prereq in resolution.prerequisites:
                prereq_id = prereq["entity_id"]
                prereq_action = prereq["action"]
                
                # Check if already executed
                if any(p["entity_id"] == prereq_id for p in executed_prerequisites):
                    continue
                
                _LOGGER.info(
                    "[IntentExecutor] Auto-enabling prerequisite: %s -> %s (for %s)",
                    prereq_action, prereq_id, entity_id
                )
                
                try:
                    domain = prereq_id.split(".")[0]
                    await self.hass.services.async_call(
                        domain,
                        prereq_action,
                        {"entity_id": prereq_id},
                    )
                    executed_prerequisites.append({
                        "entity_id": prereq_id,
                        "action": prereq_action,
                        "reason": prereq.get("reason", ""),
                        "for_entity": entity_id,
                    })
                except Exception as e:
                    _LOGGER.warning(
                        "[IntentExecutor] Failed to execute prerequisite %s: %s",
                        prereq_id, e
                    )
        
        
        # Small delay to allow devices to come online, with smart waiting
        if executed_prerequisites:
            import asyncio
            import time
            
            # Wait for up to 5 seconds for prerequisites to reach target state
            start_time = time.time()
            pending_checks = list(executed_prerequisites)
            
            while pending_checks and (time.time() - start_time) < 5.0:
                still_pending = []
                for p in pending_checks:
                    eid = p["entity_id"]
                    action = p["action"]
                    state = self.hass.states.get(eid)
                    
                    if not state:
                        continue
                        
                    # Determine target state based on action
                    is_ready = False
                    if action == "turn_on":
                        is_ready = state.state not in ("off", "unavailable", "unknown")
                    elif action == "turn_off":
                        is_ready = state.state in ("off", "unavailable")
                    else:
                        is_ready = True # Assume ready for other actions
                        
                    if not is_ready:
                        still_pending.append(p)
                
                if not still_pending:
                    break
                    
                pending_checks = still_pending
                await asyncio.sleep(0.5)
            
            _LOGGER.debug(
                "[IntentExecutor] Executed %d prerequisites: %s (Wait time: %.2fs)",
                len(executed_prerequisites),
                [p["entity_id"] for p in executed_prerequisites],
                time.time() - start_time
            )
        
        return executed_prerequisites


    async def _handle_timebox_or_delay(
        self, intent_name, eid, current_params, language, timebox_failures
    ) -> Optional[tuple]:
        """Handle TemporaryControl, TurnOn/Off with duration, and DelayedControl.
        
        Returns (effective_intent, response) tuple if handled (continue to next eid),
        or (effective_intent, None) if intent was converted but not yet handled,
        or None if not applicable.
        """
        hass = self.hass
        minutes, seconds = self._extract_duration(current_params)

        if intent_name == "TemporaryControl":
            command = current_params.get("command", "on")
            action = COMMAND_STATE_MAP.get(command, "off")

            if minutes > 0 or seconds > 0:
                success = await self._call_timebox_script(eid, minutes, seconds, action=action)
                if not success:
                    timebox_failures.append(eid)
                _LOGGER.debug(
                    "[IntentExecutor] Timebox %s on %s for %dm%ds (success=%s)",
                    action, eid, minutes, seconds, success
                )
                resp = ha_intent.IntentResponse(language=language)
                resp.response_type = ha_intent.IntentResponseType.ACTION_DONE

                state_obj = hass.states.get(eid)
                name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid

                action_de = ACTION_VERBS.get(action, action)
                duration_str = current_params.get("duration")
                if not duration_str:
                    duration_str = format_seconds_to_string(minutes * 60 + seconds)

                speech = build_confirmation(
                    "TemporaryControl",
                    [name],
                    params={"duration_str": duration_str, "action": action_de}
                )
                resp.async_set_speech(speech)
                return (intent_name, resp)
            else:
                effective = "HassTurnOn" if action == "on" else "HassTurnOff"
                return (effective, None)

        elif (intent_name in ("HassTurnOn", "HassTurnOff")) and (minutes > 0 or seconds > 0):
            action = "on" if intent_name == "HassTurnOn" else "off"
            success = await self._call_timebox_script(eid, minutes, seconds, action=action)
            if not success:
                timebox_failures.append(eid)

            resp = ha_intent.IntentResponse(language=language)
            resp.response_type = ha_intent.IntentResponseType.ACTION_DONE

            state_obj = hass.states.get(eid)
            name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid
            action_de = ACTION_VERBS.get(action, action)

            duration_str = current_params.get("duration")
            if not duration_str:
                duration_str = format_seconds_to_string(minutes * 60 + seconds)

            speech = build_confirmation(
                "TemporaryControl",
                [name],
                params={"duration_str": duration_str, "action": action_de}
            )
            resp.async_set_speech(speech)
            return (intent_name, resp)

        elif intent_name == "DelayedControl":
            delay_str = current_params.get("delay", "")
            command = current_params.get("command", "on")
            action = COMMAND_STATE_MAP.get(command, "off")

            delay_minutes, delay_seconds = self._parse_delay_or_time(delay_str)

            if delay_minutes > 0 or delay_seconds > 0:
                success = await self._call_delay_script(
                    eid, delay_minutes, delay_seconds, action=action
                )
                if not success:
                    timebox_failures.append(eid)

                _LOGGER.debug(
                    "[IntentExecutor] DelayedControl %s on %s in %dm%ds (success=%s)",
                    action, eid, delay_minutes, delay_seconds, success
                )

                resp = ha_intent.IntentResponse(language=language)
                resp.response_type = ha_intent.IntentResponseType.ACTION_DONE

                state_obj = hass.states.get(eid)
                name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid

                action_de = ACTION_VERBS.get(action, action)
                delay_display = delay_str if delay_str else format_seconds_to_string(
                    delay_minutes * 60 + delay_seconds
                )

                speech = build_confirmation(
                    "DelayedControl",
                    [name],
                    params={"delay_str": delay_display, "action": action_de}
                )
                resp.async_set_speech(speech)
                return (intent_name, resp)
            else:
                effective = "HassTurnOn" if action == "on" else "HassTurnOff"
                return (effective, None)

        return None

    def _handle_light_step(
        self, eid, current_params, final_executed_params
    ) -> Optional[str]:
        """Handle light brightness step_up/step_down adjustments.
        
        Modifies current_params and final_executed_params in place.
        Returns new effective_intent if changed (e.g. HassTurnOn for off lights), else None.
        """
        brightness_val = current_params.get("brightness") or current_params.get("command")
        if not brightness_val:
            return None
        
        val = brightness_val
        if val not in ("step_up", "step_down"):
            return None

        state_obj = self.hass.states.get(eid)
        if not state_obj:
            current_params.pop("brightness", None)
            current_params.pop("command", None)
            return None

        cur_255 = state_obj.attributes.get("brightness") or 0
        cur_pct = int((cur_255 / 255.0) * 100)
        light_is_off = state_obj.state == "off" or cur_pct == 0
        new_effective = None

        if val == "step_up":
            if light_is_off:
                new_pct = 30
                new_effective = "HassTurnOn"
                _LOGGER.debug(
                    "[IntentExecutor] step_up on OFF light %s: switching to HassTurnOn with brightness=%d%%",
                    eid, new_pct
                )
            else:
                change = max(10, int(cur_pct * self.BRIGHTNESS_STEP / 100))
                new_pct = min(100, cur_pct + change)
        else:
            change = max(10, int(cur_pct * self.BRIGHTNESS_STEP / 100))
            new_pct = max(0, cur_pct - change)

        current_params["brightness"] = new_pct
        final_executed_params["brightness"] = new_pct
        if new_pct > cur_pct or light_is_off:
            final_executed_params["direction"] = "increased"
        else:
            final_executed_params["direction"] = "decreased"
        _LOGGER.debug(
            "[IntentExecutor] %s on %s: %d%% -> %d%% (direction=%s)",
            val, eid, cur_pct, new_pct, final_executed_params["direction"]
        )
        return new_effective

    def _handle_cover_step(
        self, eid, current_params, final_executed_params
    ) -> None:
        """Handle cover step_up/step_down position adjustments.
        
        Modifies current_params and final_executed_params in place.
        """
        cmd = current_params.get("command")
        if cmd not in ("step_up", "step_down"):
            return

        state_obj = self.hass.states.get(eid)
        if state_obj:
            cur_pos = state_obj.attributes.get("current_position", 0) or 0

            if cmd == "step_up":
                new_pos = min(100, cur_pos + self.COVER_STEP)
            else:
                new_pos = max(0, cur_pos - self.COVER_STEP)

            current_params["position"] = new_pos
            current_params.pop("command", None)
            final_executed_params["position"] = new_pos
            if new_pos > cur_pos:
                final_executed_params["direction"] = "increased"
            else:
                final_executed_params["direction"] = "decreased"
            _LOGGER.debug(
                "[IntentExecutor] Cover %s on %s: %d%% -> %d%% (direction=%s)",
                cmd, eid, cur_pos, new_pos, final_executed_params["direction"]
            )
        else:
            if cmd == "step_up":
                current_params["position"] = 50
            else:
                current_params["position"] = 0
            current_params.pop("command", None)

    def _build_state_query_speech(
        self, user_input, results, entity_ids, all_entity_ids, params, language
    ) -> Optional[str]:
        """Build speech text for HassGetState / HassClimateGetTemperature queries.
        
        Returns speech text string, or None if default speech is adequate.
        """
        hass = self.hass
        
        # Collect entity data
        names = []
        states = []
        for eid, _ in results:
            state_obj = hass.states.get(eid)
            if not state_obj:
                continue
            friendly = state_obj.attributes.get("friendly_name", eid)
            names.append(friendly)
            states.append(state_obj.state)

        domain = entity_ids[0].split(".")[0] if entity_ids else None

        user_text = user_input.text.lower()
        query_state = params.get("state", "").lower()

        from ..constants.entity_keywords import LIST_QUESTION_WORDS, ALL_KEYWORDS

        is_list_question = any(w in user_text for w in LIST_QUESTION_WORDS)
        is_all_question = any(w in user_text for w in ALL_KEYWORDS)

        from ..utils.response_builder import STATE_DESCRIPTIONS_DE, build_state_response

        state_map = {
            "closed": ["closed"], "geschlossen": ["closed"],
            "open": ["open"], "offen": ["open"],
            "on": ["on"], "an": ["on"],
            "off": ["off"], "aus": ["off"],
        }
        expected_states = state_map.get(query_state, [query_state]) if query_state else []

        domain_states = STATE_DESCRIPTIONS_DE.get(domain, {})
        positive_word = domain_states.get(expected_states[0], query_state) if expected_states else ""

        opposite_word = get_opposite_state_word(positive_word)

        from ..constants.messages_de import get_state_response as _get_state_response
        from ..constants.entity_keywords import DOMAIN_NAMES_PLURAL as _DOMAIN_NAMES_PLURAL

        if is_list_question and query_state:
            plural_device = _DOMAIN_NAMES_PLURAL.get(domain, "Geräte")

            if len(names) == 0:
                return _get_state_response("none_match", device=plural_device, state=positive_word)
            elif len(names) == 1:
                return _get_state_response("state_is", device=names[0], state=positive_word)
            elif len(names) <= 5:
                return _get_state_response("states_are", devices=join_names(names), state=positive_word)
            else:
                return _get_state_response("states_are", devices=str(len(names)), state=positive_word)

        elif is_all_question and query_state:
            all_names = []
            all_states = []
            for eid in all_entity_ids:
                state_obj = hass.states.get(eid)
                if state_obj:
                    all_names.append(state_obj.attributes.get("friendly_name", eid))
                    all_states.append(state_obj.state)

            matching = [n for n, s in zip(all_names, all_states) if s in expected_states]
            not_matching = [n for n, s in zip(all_names, all_states) if s not in expected_states]

            if not not_matching:
                return CONFIRMATION_TEMPLATES["state_all_yes"].format(state=positive_word)
            else:
                if len(not_matching) == 1:
                    return CONFIRMATION_TEMPLATES["state_some_no_singular"].format(
                        name=not_matching[0], opposite=opposite_word
                    )
                elif len(not_matching) <= 3:
                    exceptions = join_names(not_matching)
                    return CONFIRMATION_TEMPLATES["state_some_no_plural"].format(
                        names=exceptions, opposite=opposite_word
                    )
                else:
                    return CONFIRMATION_TEMPLATES["state_some_no_count"].format(
                        count=len(not_matching), opposite=opposite_word
                    )

        elif query_state and len(all_entity_ids) == 1:
            entity_state = states[0] if states else hass.states.get(all_entity_ids[0]).state
            entity_name = names[0] if names else all_entity_ids[0].split(".")[-1]

            if entity_state in expected_states:
                return CONFIRMATION_TEMPLATES["state_yes_prefix"].format(
                    name=entity_name, state=positive_word
                )
            else:
                return CONFIRMATION_TEMPLATES["state_no_prefix"].format(
                    name=entity_name, opposite=opposite_word
                )

        else:
            return build_state_response(names, states, domain)

    async def run(
        self,
        user_input,
        *,
        intent_name: str,
        entity_ids: List[str],
        params: Optional[Dict[str, Any]] = None,
        language: str = "de",
        **_: Any,
    ) -> Dict[str, Any]:
        if not intent_name or not entity_ids:
            return {}

        hass = self.hass
        params = params or {}

        valid_ids = [
            eid
            for eid in entity_ids
            if hass.states.get(eid)
            and hass.states.get(eid).state not in ("unavailable", "unknown")
        ]
        if not valid_ids:
            return {}

        # --- STATE FILTERING for HassGetState queries ---
        # For "Welche Lichter sind an?" - we need to know about ALL entities
        # to properly answer "Ja, alle sind an" vs "Nein, 4 von 24 sind an"
        all_entity_ids = valid_ids.copy()  # Track ALL before filtering
        
        if intent_name == "HassGetState" and "state" in params:
            requested_state = params.get("state", "").lower()
            if requested_state:
                # Filter to entities matching the requested state
                valid_ids = [
                    eid
                    for eid in valid_ids
                    if hass.states.get(eid).state.lower() == requested_state
                ]
                _LOGGER.debug(
                    "[IntentExecutor] Filtered to %d of %d entities with state='%s'",
                    len(valid_ids),
                    len(all_entity_ids),
                    requested_state,
                )

                # If filtering results in empty list, report that
                if not valid_ids:
                    # Domain-specific device names and state words
                    domain = all_entity_ids[0].split(".")[0] if all_entity_ids else params.get("domain", "")
                    
                    from ..utils.response_builder import STATE_DESCRIPTIONS_DE
                    state_word = STATE_DESCRIPTIONS_DE.get(domain, {}).get(requested_state, requested_state)
                    
                    device_name = DOMAIN_NAMES_PLURAL.get(domain, DEFAULT_DEVICE_WORD)
                    
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    resp.async_set_speech(get_state_response("none_match", device=device_name, state=state_word))
                    return {
                        "result": ConversationResult(
                            response=resp,
                            conversation_id=user_input.conversation_id,
                            continue_conversation=False,
                        )
                    }

        # --- PHASE 2: Resolve Knowledge Graph Prerequisites ---
        # Handle power dependencies (AUTO mode) before executing main intent
        executed_prerequisites = []
        if intent_name not in ("HassGetState",):  # Skip for queries
            executed_prerequisites = await self._resolve_prerequisites(
                valid_ids, intent_name
            )

        results: List[tuple[str, ha_intent.IntentResponse]] = []
        final_executed_params = params.copy()
        final_executed_params["_prerequisites"] = executed_prerequisites  # For confirmation
        timebox_failures: List[str] = []  # Track failed timebox calls
        verification_failures: List[str] = []  # Track entities that failed verification

        for eid in valid_ids:
            effective_intent = intent_name
            domain = eid.split(".", 1)[0]
            current_params = params.copy()
            
            # --- NORMALIZE PARAMS (Fraction support) ---
            current_params = self._normalize_params(current_params)

            # --- 1. SENSOR LOGIC ---
            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                effective_intent = "HassGetState"

            # --- 2. TIMEBOX / DELAY: TemporaryControl, TurnOn/Off+duration, DelayedControl ---
            tb_result = await self._handle_timebox_or_delay(
                intent_name, eid, current_params, language, timebox_failures
            )
            if tb_result is not None:
                effective_intent, resp = tb_result
                if resp is not None:
                    results.append((eid, resp))
                    continue

            # --- 3. LIGHT LOGIC ---
            # Handle brightness from either 'brightness' or 'command' slot
            brightness_val = current_params.get("brightness") or current_params.get("command")
            
            if intent_name == "HassLightSet" and brightness_val:
                val = brightness_val

                # Timebox: if duration specified and absolute brightness
                minutes, seconds = self._extract_duration(current_params)
                if (minutes > 0 or seconds > 0) and isinstance(val, int):
                    # Call timebox with brightness value
                    await self._call_timebox_script(eid, minutes, seconds, value=val)

                    # Create fake response
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    results.append((eid, resp))
                    continue

                # Step up/down logic (RELATIVE brightness adjustments)
                new_intent = self._handle_light_step(eid, current_params, final_executed_params)
                if new_intent:
                    effective_intent = new_intent

            # --- 4. COVER: Step up/down logic (RELATIVE position adjustments) ---
            if effective_intent == "HassSetPosition":
                self._handle_cover_step(eid, current_params, final_executed_params)

            # --- 5. TIMEBOX: Cover/Fan/Climate intents ---
            minutes, seconds = self._extract_duration(current_params)
            if minutes > 0 or seconds > 0:
                value_param = None
                value = None

                # Determine which parameter contains the value
                if "position" in current_params:  # Cover
                    value_param = "position"
                    value = current_params["position"]
                elif "percentage" in current_params:  # Fan
                    value_param = "percentage"
                    value = current_params["percentage"]
                elif "temperature" in current_params:  # Climate
                    value_param = "temperature"
                    value = current_params["temperature"]

                # If we found a value to timebox
                if value is not None and isinstance(value, (int, float)):
                    await self._call_timebox_script(
                        eid, minutes, seconds, value=int(value)
                    )

                    # Create fake response
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    results.append((eid, resp))
                    continue

            # --- 6. TIMER: Handle HassTimerSet directly via service call ---
            if intent_name == "HassTimerSet":
                minutes, seconds = self._extract_duration(current_params)
                duration_sec = minutes * 60 + seconds
                
                if duration_sec > 0:
                    try:
                        await hass.services.async_call(
                            "timer", "start",
                            {"entity_id": eid, "duration": duration_sec},
                            blocking=True
                        )
                        
                        resp = ha_intent.IntentResponse(language=language)
                        resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                        
                        state_obj = hass.states.get(eid)
                        name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid
                        
                        speech = build_confirmation(
                            "HassTimerSet",
                            [name],
                            params={"duration": DURATION_TEMPLATES["minutes"].format(minutes=minutes) if minutes > 0 else DURATION_TEMPLATES["seconds"].format(seconds=seconds)}
                        )
                        resp.async_set_speech(speech)
                        results.append((eid, resp))
                        continue
                    except Exception as e:
                        _LOGGER.error("[IntentExecutor] Timer start failed for %s: %s", eid, e)
                        # Fall through to let standard handler try (or fail)

            # Slots
            slots = {"name": {"value": eid}}
            if "domain" not in current_params:
                slots["domain"] = {"value": domain}
            for k, v in current_params.items():
                if k in self.RESOLUTION_KEYS or k == "name":
                    continue
                # Skip empty string values - they cause HA intent validation errors
                if v == "" or v is None:
                    continue
                slots[k] = {"value": v}

            _LOGGER.debug("[IntentExecutor] Executing %s on %s", effective_intent, eid)

            try:
                resp = await ha_intent.async_handle(
                    hass,
                    platform="conversation",
                    intent_type=str(effective_intent),
                    slots=slots,
                    text_input=user_input.text,
                    context=user_input.context or Context(),
                    language=language or (user_input.language or "de"),
                )
                results.append((eid, resp))
                
                # Verify execution for certain intents
                if effective_intent in ("HassTurnOn", "HassTurnOff", "HassLightSet"):
                    expected_state = None
                    expected_brightness = None
                    
                    if effective_intent == "HassTurnOn":
                        expected_state = "on"
                    elif effective_intent == "HassTurnOff":
                        expected_state = "off"
                    elif effective_intent == "HassLightSet":
                        expected_state = "on"  # Light should be on after setting
                        if "brightness" in current_params:
                            expected_brightness = current_params["brightness"]
                    
                    verified = await self._verify_execution(
                        eid, effective_intent, 
                        expected_state=expected_state,
                        expected_brightness=expected_brightness
                    )
                    
                    # Track verification failures
                    if not verified:
                        verification_failures.append(eid)
                    
            except Exception as e:
                _LOGGER.warning("[IntentExecutor] Error on %s: %s", eid, e)

        if not results:
            return {}

        # If ALL timebox calls failed, return error message (but as ACTION_DONE to avoid error_code requirement)
        if timebox_failures and len(timebox_failures) == len(valid_ids):
            _LOGGER.error(
                "[IntentExecutor] All timebox calls failed for: %s", timebox_failures
            )
            resp = ha_intent.IntentResponse(language=language)
            resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
            resp.async_set_speech(ERROR_MESSAGES["timebox_failed"])
            return {
                "result": ConversationResult(
                    response=resp,
                    conversation_id=user_input.conversation_id,
                    continue_conversation=False,
                ),
                "executed_params": final_executed_params,
                "error": True,
            }

        final_resp = results[-1][1]

        # Speech Generation for State Queries
        if effective_intent in ("HassGetState", "HassClimateGetTemperature"):
            current_speech = (
                final_resp.speech.get("plain", {}).get("speech", "")
                if final_resp.speech
                else ""
            )

            if not current_speech or current_speech.strip() == SYSTEM_MESSAGES["ok"]:
                speech_text = self._build_state_query_speech(
                    user_input, results, entity_ids, all_entity_ids, params, language
                )
                if speech_text:
                    final_resp.async_set_speech(speech_text)

        def _has_speech(r):
            s = getattr(r, "speech", None)
            return isinstance(s, dict) and bool(s.get("plain", {}).get("speech"))

        if not _has_speech(final_resp):
            final_resp.async_set_speech(SYSTEM_MESSAGES["ok"])

        return {
            "result": ConversationResult(
                response=final_resp,
                conversation_id=user_input.conversation_id,
                continue_conversation=False,
            ),
            "executed_params": final_executed_params,
            "verification_failures": verification_failures,
        }

    async def _verify_execution(
        self, 
        entity_id: str, 
        intent_name: str, 
        expected_state: str = None,
        expected_brightness: int = None,
    ) -> bool:
        """Verify that an intent execution succeeded by checking entity state.
        
        Args:
            entity_id: The entity that was controlled
            intent_name: The intent that was executed
            expected_state: Expected state value (on/off) if applicable
            expected_brightness: Expected brightness percentage if applicable
            
        Returns:
            True if verification passed, False if there's a mismatch
        """
        # Domain-specific verification timeouts (seconds)
        domain = entity_id.split(".")[0]
        timeout_map = {
            "media_player": 10.0,  # Radios need boot time
            "climate": 5.0,        # HVAC can be slow
            "vacuum": 5.0,         # Vacuums can be slow
        }
        max_wait = timeout_map.get(domain, 2.0)  # Default 2 seconds
        
        import time
        start_time = time.time()
        last_state = None
        
        # Poll for state change
        while (time.time() - start_time) < max_wait:
            await asyncio.sleep(0.5)
            
            state = self.hass.states.get(entity_id)
            if not state:
                _LOGGER.warning("[IntentExecutor] Verification: entity %s not found", entity_id)
                return False
            
            current = state.state.lower()
            last_state = current
            
            # Check if state matches expected
            if expected_state:
                expected = expected_state.lower()
                
                # Special handling for covers
                if entity_id.startswith("cover."):
                    if expected == "on":
                        expected = "open"
                    elif expected == "off":
                        expected = "closed"
                    
                    # Accept transitional states
                    if expected == "open" and current in ("open", "opening"):
                        _LOGGER.debug("[IntentExecutor] Verification passed for %s (state: %s)", entity_id, current)
                        return True
                    elif expected == "closed" and current in ("closed", "closing"):
                        _LOGGER.debug("[IntentExecutor] Verification passed for %s (state: %s)", entity_id, current)
                        return True
                
                # Standard entities and media players
                elif current == expected:
                    _LOGGER.debug("[IntentExecutor] Verification passed for %s after %.1fs", entity_id, time.time() - start_time)
                    return True
                
                # For media_player, also accept "playing", "paused", "idle" as "on" states
                elif domain == "media_player" and expected == "on" and current not in ("off", "unavailable", "unknown"):
                    _LOGGER.debug("[IntentExecutor] Verification passed for %s (media state: %s)", entity_id, current)
                    return True
                
                # For scenes, state is a timestamp of last activation
                # Scene states look like: "2025-12-25t17:39:38.947016+00:00"
                # Verify the timestamp is recent (within last 10 seconds)
                elif domain == "scene" and expected == "on":
                    try:
                        import homeassistant.util.dt as dt_util
                        # Parse ISO format timestamp - use raw state to preserve T separator
                        scene_time = dt_util.parse_datetime(state.state.replace("Z", "+00:00"))
                        now = dt_util.utcnow()
                        age_seconds = (now - scene_time).total_seconds()
                        _LOGGER.debug("[IntentExecutor] Scene verification: current='%s', now='%s', age=%.1fs", state.state, now, age_seconds)
                        if age_seconds < 10:  # Activated within last 10 seconds
                            _LOGGER.debug("[IntentExecutor] Verification passed for %s (scene activated %.1fs ago)", entity_id, age_seconds)
                            return True
                        else:
                            _LOGGER.debug("[IntentExecutor] Scene %s last activated %.1fs ago (too old)", entity_id, age_seconds)
                    except (ValueError, TypeError) as e:
                        _LOGGER.debug("[IntentExecutor] Could not parse scene timestamp '%s': %s", current, e)
            

            # Check brightness if applicable
            if expected_brightness is not None:
                cur_255 = state.attributes.get("brightness") or 0
                cur_pct = int((cur_255 / 255.0) * 100)
                if abs(cur_pct - expected_brightness) <= 5:
                    _LOGGER.debug("[IntentExecutor] Verification passed for %s (brightness: %d%%)", entity_id, cur_pct)
                    return True
        
        # Timeout - log failure
        _LOGGER.warning(
            "[IntentExecutor] Verification FAILED for %s after %.1fs: expected '%s', got '%s'",
            entity_id, max_wait, expected_state, last_state
        )
        return False

