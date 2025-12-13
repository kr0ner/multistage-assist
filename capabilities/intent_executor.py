import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import intent as ha_intent
from homeassistant.core import Context
from homeassistant.components.conversation import ConversationResult

from custom_components.multistage_assist.conversation_utils import (
    join_names,
    normalize_speech_for_tts,
    parse_duration_string,
)
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentExecutorCapability(Capability):
    """Execute a known HA intent for one or more concrete entity_ids."""

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets."

    RESOLUTION_KEYS = {"area", "floor", "name", "entity_id"}
    BRIGHTNESS_STEP = 20  # Percentage to change for step_up/step_down
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
            await self.hass.services.async_call(
                "script", "timebox_entity_state", data, blocking=True
            )
            return True
        except Exception as e:
            _LOGGER.error(
                "[IntentExecutor] Timebox script failed for %s: %s", entity_id, e
            )
            return False

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
                    "[IntentExecutor] Filtered to %d entities with state='%s'",
                    len(valid_ids),
                    requested_state,
                )

                # Try yes/no response capability
                yes_no_cap = self.get("yes_no_response")
                if yes_no_cap:
                    response_text = await yes_no_cap.run(
                        user_input,
                        domain=params.get("domain", ""),
                        state=requested_state,
                        entity_ids=valid_ids,
                    )

                    if response_text:
                        # Yes/no question detected - return boolean answer
                        resp = ha_intent.IntentResponse(language=language)
                        resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                        resp.async_set_speech(response_text)
                        return {"result": resp}

                # Not a yes/no question or no capability - check if empty
                if not valid_ids:
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    resp.async_set_speech("Es gibt keine passenden Geräte.")
                    return {"result": resp}

        results: List[tuple[str, ha_intent.IntentResponse]] = []
        final_executed_params = params.copy()
        timebox_failures: List[str] = []  # Track failed timebox calls

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

                # Create fake response
                resp = ha_intent.IntentResponse(language=language)
                resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
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

                # Step up/down logic (relative brightness adjustments)
                if val in ("step_up", "step_down"):
                    state_obj = hass.states.get(eid)
                    if state_obj:
                        cur_255 = state_obj.attributes.get("brightness") or 0
                        cur_pct = int((cur_255 / 255.0) * 100)

                        if val == "step_up":
                            new_pct = min(100, cur_pct + self.BRIGHTNESS_STEP)
                            if cur_pct == 0:
                                new_pct = self.BRIGHTNESS_STEP
                        else:
                            new_pct = max(0, cur_pct - self.BRIGHTNESS_STEP)

                        current_params["brightness"] = new_pct
                        final_executed_params["brightness"] = new_pct
                        _LOGGER.debug(
                            "[IntentExecutor] %s on %s: %d%% -> %d%%",
                            val, eid, cur_pct, new_pct
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
            except Exception as e:
                _LOGGER.warning("[IntentExecutor] Error on %s: %s", eid, e)

        if not results:
            return {}

        # If ALL timebox calls failed, return error
        if timebox_failures and len(timebox_failures) == len(valid_ids):
            _LOGGER.error(
                "[IntentExecutor] All timebox calls failed for: %s", timebox_failures
            )
            resp = ha_intent.IntentResponse(language=language)
            resp.response_type = ha_intent.IntentResponseType.ERROR
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

        # Speech Generation
        if effective_intent in ("HassGetState", "HassClimateGetTemperature"):
            current_speech = (
                final_resp.speech.get("plain", {}).get("speech", "")
                if final_resp.speech
                else ""
            )

            if not current_speech or current_speech.strip() == "Okay":
                parts = []
                for eid, _ in results:
                    state_obj = hass.states.get(eid)

                    if not state_obj:
                        continue

                    friendly = state_obj.attributes.get("friendly_name", eid)
                    val = state_obj.state
                    unit = state_obj.attributes.get("unit_of_measurement", "")

                    # Translate common English states to German
                    if language == "de":
                        state_translations = {
                            "off": "aus",
                            "on": "an",
                            "open": "offen",
                            "closed": "geschlossen",
                            "locked": "verschlossen",
                            "unlocked": "aufgeschlossen",
                        }
                        val = state_translations.get(val.lower(), val)

                    if val.replace(".", "", 1).isdigit():
                        val = val.replace(".", ",")
                    text_part = f"{friendly} ist {val}"

                    if unit:
                        text_part += f" {unit}"

                    parts.append(text_part)

                if parts:
                    raw_text = join_names(parts) + "."
                    speech_text = normalize_speech_for_tts(raw_text)
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
        }
