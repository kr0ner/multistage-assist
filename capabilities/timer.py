import logging
import re
import asyncio
from typing import Any, Dict, List, Optional
import importlib

from .base import Capability
from custom_components.multistage_assist.conversation_utils import make_response, parse_duration_string, format_seconds_to_string

_LOGGER = logging.getLogger(__name__)

_fuzz = None
async def _get_fuzz():
    global _fuzz
    if _fuzz is not None: return _fuzz
    loop = asyncio.get_running_loop()
    def _load(): return importlib.import_module("rapidfuzz.fuzz")
    _fuzz = await loop.run_in_executor(None, _load)
    return _fuzz

class TimerCapability(Capability):
    name = "timer"
    description = "Set timers on mobile devices."

    async def run(self, user_input, intent_name: str, slots: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        # Accept both intents
        if intent_name not in ("HassTimerSet", "HassStartTimer"):
            return {}

        duration_raw = slots.get("duration")
        # Stage0 NLU might use "minutes" slot instead of "duration"
        if not duration_raw:
             if slots.get("minutes"): duration_raw = str(slots.get("minutes")) + " Minuten"
             if slots.get("seconds"): duration_raw = str(slots.get("seconds")) + " Sekunden"
        
        device_name = slots.get("name")
        device_id = slots.get("device_id") 
        return await self._process_request(user_input, duration_raw, device_name, device_id)

    async def continue_flow(self, user_input, pending_data: Dict[str, Any]) -> Dict[str, Any]:
        step = pending_data.get("step")
        # Restore state
        duration = pending_data.get("duration")
        device_id = pending_data.get("device_id")
        device_name = pending_data.get("name") # Original query name
        
        text = user_input.text
        learning_data = None

        if step == "ask_duration":
            seconds = parse_duration_string(text)
            if not seconds:
                # If still invalid, ask again
                return {
                    "status": "handled",
                    "result": await make_response("Ich habe die Zeit nicht verstanden. Bitte sag z.B. '5 Minuten'.", user_input),
                    "pending_data": pending_data
                }
            duration = seconds # Now we have duration (in seconds)

        elif step == "ask_device":
            candidates = pending_data.get("candidates", [])
            matched = await self._fuzzy_match_device(text, candidates)
            if not matched:
                return {
                    "status": "handled",
                    "result": await make_response("Das habe ich nicht verstanden. Welches Gerät?", user_input),
                    "pending_data": pending_data
                }
            device_id = matched
            
            # If we had an original name that failed to match automatically, and user now selected one manually -> LEARN IT
            if device_name:
                learning_data = {
                    "type": "entity",
                    "source": device_name,
                    "target": device_id
                }

        # Recursively call process request to check if we have everything now
        # Pass what we have. If something is still missing, it will ask for the next thing.
        res = await self._process_request(user_input, duration, device_name=device_name, device_id=device_id)
        
        # Inject learning data if we just resolved the device
        if learning_data:
            res["learning_data"] = learning_data
            
        return res

    async def _process_request(self, user_input, duration_raw, device_name=None, device_id=None) -> Dict[str, Any]:
        # 1. Resolve Duration
        # If passed as int (already parsed), use it. If str, parse it.
        if isinstance(duration_raw, int):
            seconds = duration_raw
        else:
            seconds = parse_duration_string(duration_raw) if duration_raw else 0
        
        # ASK FOR DURATION if missing
        if not seconds:
            return {
                "status": "handled", 
                "result": await make_response("Wie lange soll der Timer laufen?", user_input), 
                "pending_data": {
                    "type": "timer", 
                    "step": "ask_duration", 
                    "device_id": device_id, 
                    "name": device_name
                }
            }

        # 2. Resolve Device
        if not device_id:
            mobile_services = self._get_mobile_notify_services()
            if not mobile_services:
                return {"status": "handled", "result": await make_response("Keine mobilen Geräte gefunden.", user_input)}
            
            # Try fuzzy match on initial name
            if device_name:
                device_id = await self._fuzzy_match_device(device_name, mobile_services)
            
            # ASK FOR DEVICE if still missing
            if not device_id:
                if len(mobile_services) == 1:
                    device_id = mobile_services[0]["service"]
                else:
                    names = [d["name"] for d in mobile_services]
                    return {
                        "status": "handled", 
                        "result": await make_response(f"Auf welchem Gerät? ({', '.join(names)})", user_input), 
                        "pending_data": {
                            "type": "timer", 
                            "step": "ask_device", 
                            "duration": seconds, # Pass resolved duration forward
                            "candidates": mobile_services, 
                            "name": device_name # Keep original name for learning
                        }
                    }

        # 3. Execute
        await self._set_android_timer(device_id, seconds)
        
        # Get friendly name
        device_friendly = device_id.split(".")[-1].replace("mobile_app_", "").replace("_", " ").title()
        services = self._get_mobile_notify_services()
        for s in services:
            if s["service"] == device_id:
                device_friendly = s["name"]
                break

        return {
            "status": "handled",
            "result": await make_response(f"Timer für {format_seconds_to_string(seconds)} auf {device_friendly} gestellt.", user_input)
        }

    def _get_mobile_notify_services(self) -> List[Dict[str, str]]:
        services = self.hass.services.async_services().get("notify", {})
        return [{"service": f"notify.{k}", "name": k.replace("mobile_app_", "").replace("_", " ").title()} for k in services if k.startswith("mobile_app_")]

    async def _fuzzy_match_device(self, query: str, candidates: List[Dict[str, str]]) -> Optional[str]:
        if not query: return None
        fuzz = await _get_fuzz()
        best_score = 0
        best_id = None
        q = query.lower().strip()
        for c in candidates:
            name_score = fuzz.token_set_ratio(q, c["name"].lower())
            id_score = fuzz.token_set_ratio(q, c["service"].split(".")[-1])
            score = max(name_score, id_score)
            if score > best_score and score > 70:
                best_score = score
                best_id = c["service"]
        return best_id

    async def _set_android_timer(self, service_full: str, seconds: int):
        domain, service = service_full.split(".", 1)
        payload = {"message": "command_activity", "data": {"intent_action": "android.intent.action.SET_TIMER", "intent_extras": f"android.intent.extra.alarm.LENGTH:{seconds},android.intent.extra.alarm.SKIP_UI:true"}}
        await self.hass.services.async_call(domain, service, payload)