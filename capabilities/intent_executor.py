import logging
from typing import Any, Dict, List, Optional

from homeassistant.helpers import intent as ha_intent
from homeassistant.core import Context
from homeassistant.components.conversation import ConversationResult

from .base import Capability

_LOGGER = logging.getLogger(__name__)


def _join_names(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} und {names[1]}"
    return f"{', '.join(names[:-1])} und {names[-1]}"


class IntentExecutorCapability(Capability):
    """
    Execute a known HA intent for one or more concrete entity_ids.
    Does NOT provide generic fallback speech anymore; leaves that to ResponseGenerator.
    """

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets."

    # Keys that are used for resolving entities but should NOT be passed to execution
    RESOLUTION_KEYS = {"area", "room", "floor", "device_class"}

    # Step size for relative brightness
    BRIGHTNESS_STEP = 20

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
            _LOGGER.warning("[IntentExecutor] Missing intent/entities.")
            return {}

        hass = self.hass
        params = params or {}

        # 1. Validate entities
        valid_ids = [eid for eid in entity_ids if hass.states.get(eid) and hass.states.get(eid).state not in ("unavailable", "unknown")]
        
        if not valid_ids:
            return {}

        # 2. Execute intent for each entity
        results: List[tuple[str, ha_intent.IntentResponse]] = []
        
        for eid in valid_ids:
            effective_intent = intent_name
            domain = eid.split(".", 1)[0]
            current_params = params.copy()

            # Downgrade Climate intent for Sensors
            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                effective_intent = "HassGetState"

            # Relative Brightness Logic
            if intent_name == "HassLightSet" and "brightness" in current_params:
                val = current_params["brightness"]
                if val in ("step_up", "step_down"):
                    state_obj = hass.states.get(eid)
                    if state_obj:
                        current_level_255 = state_obj.attributes.get("brightness") or 0
                        current_pct = int((current_level_255 / 255.0) * 100)
                        
                        if val == "step_up":
                            new_pct = min(100, current_pct + self.BRIGHTNESS_STEP)
                            # Turn on if off
                            if current_pct == 0: new_pct = self.BRIGHTNESS_STEP
                        else:
                            new_pct = max(0, current_pct - self.BRIGHTNESS_STEP)
                        
                        current_params["brightness"] = new_pct
                    else:
                        current_params.pop("brightness")

            # Prepare slots (exclude resolution keys)
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

        # 3. Custom Speech for Queries (Readouts)
        # We only inject speech if it's a query intent. 
        # For actions (TurnOn/Off), we leave it empty so Stage1 can use ResponseGenerator.
        if effective_intent in ("HassGetState", "HassClimateGetTemperature"):
            current_speech = final_resp.speech.get("plain", {}).get("speech", "") if final_resp.speech else ""
            
            if not current_speech or current_speech.strip() == "Okay":
                parts = []
                for eid, _ in results:
                    state_obj = hass.states.get(eid)
                    if not state_obj: continue
                    friendly = state_obj.attributes.get("friendly_name", eid)
                    val = state_obj.state
                    unit = state_obj.attributes.get("unit_of_measurement", "")
                    text_part = f"{friendly} ist {val}"
                    if unit: text_part += f" {unit}"
                    parts.append(text_part)
                
                if parts:
                    final_resp.async_set_speech(_join_names(parts) + ".")

        conv_result = ConversationResult(
            response=final_resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )
        return {"result": conv_result}
