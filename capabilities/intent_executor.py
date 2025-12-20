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
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentExecutorCapability(Capability):
    """Execute a known HA intent for one or more concrete entity_ids."""

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets."

    RESOLUTION_KEYS = {"area", "floor", "name", "entity_id"}
    BRIGHTNESS_STEP = 35  # Percentage of current brightness for step_up/step_down
    TIMEBOX_SCRIPT_ENTITY_ID = "script.timebox_entity_state"

    def _extract_duration(self, params: Dict[str, Any]) -> tuple[int, int]:
        """Extract minutes and seconds from params. Returns (minutes, seconds)."""
        duration_raw = params.get("duration")
        if duration_raw:
            seconds = parse_duration_string(duration_raw)
            return (seconds // 60, seconds % 60)
        return (0, 0)

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
    }
    
    async def _resolve_prerequisites(
        self, entity_ids: List[str], intent_name: str
    ) -> List[Dict[str, Any]]:
        """Resolve and execute Knowledge Graph prerequisites for entities.
        
        This handles power dependencies and device coupling with AUTO mode.
        
        Args:
            entity_ids: Entities to check
            intent_name: Intent being executed
            
        Returns:
            List of prerequisites that were executed
        """
        try:
            from ..utils.knowledge_graph import get_knowledge_graph
        except ImportError:
            return []
        
        action = self.INTENT_TO_ACTION.get(intent_name, "turn_on")
        graph = get_knowledge_graph(self.hass)
        
        executed_prerequisites = []
        
        for entity_id in entity_ids:
            resolution = graph.resolve_for_action(entity_id, action)
            
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
        print(f"DEBUG: IntentExecutor.run called with user_input: {user_input}")
        print(f"DEBUG: intent_name: {intent_name}")
        print(f"DEBUG: entity_ids: {entity_ids}")
        print(f"DEBUG: params: {params}")
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
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    resp.async_set_speech(f"Keine Geräte sind {requested_state}.")
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

            # --- 1. SENSOR LOGIC ---
            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                effective_intent = "HassGetState"

            # --- 2. TIMEBOX: HassTemporaryControl or HassTurnOn/Off with duration ---
            minutes, seconds = self._extract_duration(current_params)
            
            # Handle HassTemporaryControl (convert to timebox)
            if intent_name == "HassTemporaryControl":
                command = current_params.get("command", "on")
                action = "on" if command in ("on", "an", "ein", "auf") else "off"
                
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
                    
                    # Generate confirmation speech
                    state_obj = hass.states.get(eid)
                    name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid
                    
                    action_de = "an" if action == "on" else "aus"
                    duration_str = current_params.get("duration")
                    if not duration_str:
                        duration_str = format_seconds_to_string(minutes * 60 + seconds)

                    speech = build_confirmation(
                        "HassTemporaryControl",
                        [name],
                        params={"duration_str": duration_str, "action": action_de}
                    )
                    resp.async_set_speech(speech)
                    
                    results.append((eid, resp))
                    continue
                else:
                    # No duration - convert to regular on/off
                    effective_intent = "HassTurnOn" if action == "on" else "HassTurnOff"
            
            # Handle HassTurnOn/Off with duration (legacy path)
            elif (intent_name == "HassTurnOn" or intent_name == "HassTurnOff") and (
                minutes > 0 or seconds > 0
            ):
                action = "on" if intent_name == "HassTurnOn" else "off"
                success = await self._call_timebox_script(eid, minutes, seconds, action=action)
                if not success:
                    timebox_failures.append(eid)

                # Create response with speech
                resp = ha_intent.IntentResponse(language=language)
                resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                
                state_obj = hass.states.get(eid)
                name = state_obj.attributes.get("friendly_name", eid) if state_obj else eid
                action_de = "an" if action == "on" else "aus"
                
                duration_str = current_params.get("duration")
                if not duration_str:
                    duration_str = format_seconds_to_string(minutes * 60 + seconds)

                speech = build_confirmation(
                    "HassTemporaryControl",
                    [name],
                    params={"duration_str": duration_str, "action": action_de}
                )
                resp.async_set_speech(speech)
                
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
                # step_up: increase by BRIGHTNESS_STEP% of current (e.g., 50% -> 60% if step=20%)
                # step_down: decrease by BRIGHTNESS_STEP% of current (e.g., 50% -> 40% if step=20%)
                if val in ("step_up", "step_down"):
                    state_obj = hass.states.get(eid)
                    if state_obj:
                        cur_255 = state_obj.attributes.get("brightness") or 0
                        cur_pct = int((cur_255 / 255.0) * 100)

                        if val == "step_up":
                            if cur_pct == 0:
                                # Light is off, turn on to reasonable brightness
                                new_pct = 30
                            else:
                                # Increase by BRIGHTNESS_STEP% of current, minimum 10%
                                change = max(10, int(cur_pct * self.BRIGHTNESS_STEP / 100))
                                new_pct = min(100, cur_pct + change)
                        else:
                            # step_down: reduce by BRIGHTNESS_STEP% of current, minimum 10%
                            change = max(10, int(cur_pct * self.BRIGHTNESS_STEP / 100))
                            new_pct = max(0, cur_pct - change)

                        current_params["brightness"] = new_pct
                        final_executed_params["brightness"] = new_pct
                        _LOGGER.debug(
                            "[IntentExecutor] %s on %s: %d%% -> %d%% (change: ±%d%%)",
                            val, eid, cur_pct, new_pct, change if cur_pct > 0 else 30
                        )
                    else:
                        current_params.pop("brightness", None)
                        current_params.pop("command", None)

            # --- 4. TIMEBOX: Cover/Fan/Climate intents ---
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

            # Slots
            slots = {"name": {"value": eid}}
            if "domain" not in current_params:
                slots["domain"] = {"value": domain}
            for k, v in current_params.items():
                if k in self.RESOLUTION_KEYS or k == "name":
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
            resp.async_set_speech("Fehler beim Ausführen der zeitlichen Steuerung.")
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

            if not current_speech or current_speech.strip() == "Okay":
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
                
                # Detect question type from user input
                user_text = user_input.text.lower()
                query_state = params.get("state", "").lower()
                
                # "Welche Lichter sind an?" → LIST the matching ones
                # "Sind alle Lichter an?" → YES/NO about ALL entities
                is_list_question = any(w in user_text for w in ["welche", "welches", "was für", "was ist", "wie viele"])
                is_all_question = "alle" in user_text or "sämtliche" in user_text
                
                from ..utils.response_builder import STATE_DESCRIPTIONS_DE, build_state_response
                
                # State mapping for comparison
                state_map = {
                    "closed": ["closed"],
                    "geschlossen": ["closed"],
                    "open": ["open"],
                    "offen": ["open"],
                    "on": ["on"],
                    "an": ["on"],
                    "off": ["off"],
                    "aus": ["off"],
                }
                expected_states = state_map.get(query_state, [query_state]) if query_state else []
                
                # Get German state words
                domain_states = STATE_DESCRIPTIONS_DE.get(domain, {})
                positive_word = domain_states.get(expected_states[0], query_state) if expected_states else ""
                opposite_map = {"an": "aus", "aus": "an", "offen": "geschlossen", "geschlossen": "offen"}
                opposite_word = opposite_map.get(positive_word, "anders")
                
                if is_list_question and query_state:
                    # LIST QUESTION: "Welche Lichter sind an?" → list the matching ones (names)
                    # names/states come from filtered results (only matching entities)
                    if len(names) == 0:
                        speech_text = f"Keine sind {positive_word}."
                    elif len(names) == 1:
                        speech_text = f"{names[0]} ist {positive_word}."
                    elif len(names) <= 5:
                        speech_text = f"{join_names(names)} sind {positive_word}."
                    else:
                        speech_text = f"{len(names)} sind {positive_word}."
                        
                elif is_all_question and query_state:
                    # YES/NO QUESTION: "Sind alle Lichter an?" → check ALL entities
                    # Get names and states for ALL entities (not just filtered)
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
                        # All match - simple "Ja"
                        speech_text = f"Ja, alle sind {positive_word}."
                    else:
                        # Some don't match - list the NOT matching ones
                        if len(not_matching) <= 3:
                            exceptions = join_names(not_matching)
                            verb = "ist" if len(not_matching) == 1 else "sind"
                            speech_text = f"Nein, {exceptions} {verb} noch {opposite_word}."
                        else:
                            speech_text = f"Nein, {len(not_matching)} sind noch {opposite_word}."
                
                elif query_state and len(all_entity_ids) == 1:
                    # SINGLE ENTITY YES/NO: "Ist das Licht in der Dusche an?"
                    # → "Ja, Dusche ist an." / "Nein, Dusche ist aus."
                    entity_state = states[0] if states else hass.states.get(all_entity_ids[0]).state
                    entity_name = names[0] if names else all_entity_ids[0].split(".")[-1]
                    
                    if entity_state in expected_states:
                        speech_text = f"Ja, {entity_name} ist {positive_word}."
                    else:
                        speech_text = f"Nein, {entity_name} ist {opposite_word}."
                
                else:
                    # Normal state query - use template
                    speech_text = build_state_response(names, states, domain)
                
                final_resp.async_set_speech(speech_text)

        def _has_speech(r):
            s = getattr(r, "speech", None)
            return isinstance(s, dict) and bool(s.get("plain", {}).get("speech"))

        if not _has_speech(final_resp):
            final_resp.async_set_speech("Okay.")

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

