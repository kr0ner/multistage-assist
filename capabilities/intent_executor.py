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

    RESOLUTION_KEYS = {"area", "room", "floor", "device_class"}
    BRIGHTNESS_STEP = 20
    SCRIPT_ENTITY_ID = "script.temporary_control_generic"

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

        # --- SAFETY: PREVENT MASS TEMP CONTROL ---
        if intent_name == "HassTemporaryControl" and len(valid_ids) > 5:
            _LOGGER.warning(
                "[IntentExecutor] Aborting Temporary Control: Too many targets (%d).",
                len(valid_ids),
            )
            return {}
        # -----------------------------------------

        results: List[tuple[str, ha_intent.IntentResponse]] = []
        final_executed_params = params.copy()

        for eid in valid_ids:
            effective_intent = intent_name
            domain = eid.split(".", 1)[0]
            current_params = params.copy()

            # --- 1. SENSOR LOGIC ---
            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                effective_intent = "HassGetState"

            # --- 2. LIGHT LOGIC ---
            if intent_name == "HassLightSet" and "brightness" in current_params:
                val = current_params["brightness"]
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
                    else:
                        current_params.pop("brightness")

            # --- 3. TEMPORARY CONTROL LOGIC ---
            if intent_name == "HassTemporaryControl":
                # Verify script
                if not hass.states.get(self.SCRIPT_ENTITY_ID):
                    _LOGGER.error("Script %s not found!", self.SCRIPT_ENTITY_ID)
                    continue

                duration_raw = current_params.get("duration")
                raw_command = current_params.get("command", "on")
                command = "on"
                if raw_command.lower() in ("aus", "off", "false", "0", "zu"):
                    command = "off"

                seconds = parse_duration_string(duration_raw)
                if seconds < 1:
                    seconds = 10

                _LOGGER.debug(
                    "[IntentExecutor] Running script for %s: %s for %d seconds",
                    eid,
                    command,
                    seconds,
                )
                # Execute
                try:
                    import asyncio

                    service_domain = "script"
                    service_name = "temporary_control_generic"
                    service_data = {
                        "target_entity": eid,
                        "seconds": seconds,
                        "command": command,
                    }
                    call_context = Context()

                    print(
                        f"DEBUG: Calling async_call with {service_domain}, {service_name}, {service_data}"
                    )
                    print(
                        f"DEBUG: async_call type: {type(self.hass.services.async_call)}"
                    )
                    print(
                        f"DEBUG: async_call is_coroutine: {asyncio.iscoroutinefunction(self.hass.services.async_call)}"
                    )

                    # Force return value to be awaitable if it's a mock and not configured correctly
                    # But it should be AsyncMock.

                    ret = self.hass.services.async_call(
                        service_domain,
                        service_name,
                        service_data,
                        blocking=False,  # Non-blocking so we don't wait for the delay
                        context=call_context,
                    )
                    print(f"DEBUG: async_call returned: {ret}, type: {type(ret)}")

                    if asyncio.iscoroutine(ret) or hasattr(ret, "__await__"):
                        await ret
                    else:
                        print("DEBUG: Return value is not awaitable!")
                        print(f"DEBUG: async_call returned: {ret}, type: {type(ret)}")

                    # Fake a response for feedback
                    resp = ha_intent.IntentResponse(language=language)
                    resp.response_type = ha_intent.IntentResponseType.ACTION_DONE
                    results.append((eid, resp))
                    continue
                except Exception as e:
                    _LOGGER.error("Failed to call temporary control script: %s", e)
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
