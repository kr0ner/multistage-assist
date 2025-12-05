import logging
import re
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


def _normalize_speech(text: str) -> str:
    """Normalize text for German TTS (Piper)."""
    if not text:
        return ""
    
    # 1. Replace decimal dots with commas (e.g. 22.5 -> 22,5)
    # Matches a digit, followed by a dot, followed by a digit
    text = re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)
    
    # 2. Expand common units
    replacements = {
        "°C": " Grad Celsius",
        "°": " Grad",
        "%": " Prozent",
        "kWh": " Kilowattstunden",
        "kW": " Kilowatt",
        "W": " Watt",
        "V": " Volt",
        "A": " Ampere",
        "lx": " Lux",
        "lm": " Lumen",
    }
    
    for symbol, spoken in replacements.items():
        # Replace symbol if it's at end of string or followed by space/punctuation
        # This prevents replacing "V" inside "Volumen"
        text = re.sub(rf"{re.escape(symbol)}(?=$|\s|[.,!?])", spoken, text)
        
    return text.strip()


class IntentExecutorCapability(Capability):
    """
    Execute a known HA intent for one or more concrete entity_ids by calling
    homeassistant.helpers.intent.async_handle directly.
    """

    name = "intent_executor"
    description = "Execute a Home Assistant intent for specific targets and return a ConversationResult."

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
            _LOGGER.warning(
                "[IntentExecutor] Missing intent_name or entity_ids (intent=%r, entities=%r)",
                intent_name, entity_ids,
            )
            return {}

        hass = self.hass
        params = params or {}

        # 1. Validate entities (skip unavailable)
        valid_ids = []
        for eid in entity_ids:
            st = hass.states.get(eid)
            if st and st.state not in ("unavailable", "unknown"):
                valid_ids.append(eid)
            else:
                _LOGGER.warning("[IntentExecutor] Skipping unavailable entity: %s", eid)

        if not valid_ids:
            return {}

        # 2. Execute intent for each entity
        results: List[tuple[str, ha_intent.IntentResponse]] = []
        
        for eid in valid_ids:
            # Determine effective intent and params
            effective_intent = intent_name
            domain = eid.split(".", 1)[0]
            current_params = params.copy()

            # --- Logic for Sensors (Downgrade Climate intent) ---
            if intent_name == "HassClimateGetTemperature" and domain == "sensor":
                _LOGGER.debug("[IntentExecutor] Downgrading %s to HassGetState for sensor %s", intent_name, eid)
                effective_intent = "HassGetState"

            # --- Logic for Relative Brightness (Light domain) ---
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

            # Prepare slots
            slots = {"name": {"value": eid}}
            
            if "domain" not in current_params:
                slots["domain"] = {"value": domain}

            # Add processed params to slots, skipping resolution constraints
            for k, v in current_params.items():
                if k in self.RESOLUTION_KEYS or k == "name":
                    continue
                slots[k] = {"value": v}

            _LOGGER.debug(
                "[IntentExecutor] Executing intent '%s' on %s with slots=%s",
                effective_intent, eid, slots,
            )

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
                _LOGGER.warning("[IntentExecutor] execution failed for %s: %s", eid, e)

        if not results:
            _LOGGER.warning("[IntentExecutor] No successful responses collected.")
            return {}

        # Use the last response as the base for the result object
        final_resp = results[-1][1]

        # 3. Custom Speech Generation for Queries
        if effective_intent in ("HassGetState", "HassClimateGetTemperature"):
            current_speech = ""
            if final_resp.speech:
                current_speech = final_resp.speech.get("plain", {}).get("speech", "")
            
            if not current_speech or current_speech.strip() == "Okay":
                parts = []
                for eid, _ in results:
                    state_obj = hass.states.get(eid)
                    if not state_obj:
                        continue
                    
                    friendly = state_obj.attributes.get("friendly_name", eid)
                    val = state_obj.state
                    unit = state_obj.attributes.get("unit_of_measurement", "")
                    
                    # Normalize the value (dot to comma)
                    if val.replace(".", "", 1).isdigit():
                        val = val.replace(".", ",")
                    
                    text_part = f"{friendly} ist {val}"
                    if unit:
                        text_part += f" {unit}"
                    parts.append(text_part)
                
                if parts:
                    raw_text = _join_names(parts) + "."
                    # Normalize the full text (units, etc.)
                    speech_text = _normalize_speech(raw_text)
                    
                    final_resp.async_set_speech(speech_text)
                    _LOGGER.debug("[IntentExecutor] Injected normalized speech: %s", speech_text)

        # 4. General Fallback
        def _has_plain_speech(r: ha_intent.IntentResponse) -> bool:
            s = getattr(r, "speech", None)
            if not isinstance(s, dict):
                return False
            plain = s.get("plain") or {}
            return bool(plain.get("speech"))

        if not _has_plain_speech(final_resp):
            final_resp.async_set_speech("Okay.")

        conv_result = ConversationResult(
            response=final_resp,
            conversation_id=user_input.conversation_id,
            continue_conversation=False,
        )
        return {"result": conv_result}
