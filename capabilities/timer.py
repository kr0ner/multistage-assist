import logging
import asyncio
from typing import Any, Dict, List, Optional
from homeassistant.core import HomeAssistant
from homeassistant.components import conversation

from .base import Capability
from custom_components.multistage_assist.conversation_utils import (
    make_response,
    parse_duration_string,
    format_seconds_to_string,
)

from ..utils.fuzzy_utils import get_fuzz, fuzzy_match_best


_LOGGER = logging.getLogger(__name__)


class TimerCapability(Capability):
    name = "timer"
    description = "Set timers on mobile devices."

    # Prompt to extract timer description from natural language
    PROMPT = {
        "system": """Extract a short, descriptive timer label from the user's request.
If the user mentions what the timer is for (e.g., "remind me pasta is done", "Nudeln fertig", "Pizza aus dem Ofen"), 
extract a 2-3 word label. Otherwise return empty.

Examples:
"Setze einen Timer für 5 Minuten der mich daran erinnert, dass die Nudeln fertig sind" → {"description": "Nudeln"}
"Timer für 10 Minuten damit die Pizza nicht verbrennt" → {"description": "Pizza"}
"5 Minuten Timer für den Tee" → {"description": "Tee"}
"Timer für 3 Minuten" → {"description": ""}
"Stelle einen Timer auf 20 Minuten" → {"description": ""}
""",
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
        },
    }

    async def run(
        self, user_input, intent_name: str, slots: Dict[str, Any], **_: Any
    ) -> Dict[str, Any]:
        # Accept both intents
        if intent_name not in ("HassTimerSet", "HassStartTimer"):
            return {}

        duration_raw = slots.get("duration")
        # Stage0 NLU might use "minutes" slot instead of "duration"
        if not duration_raw:
            if slots.get("minutes"):
                duration_raw = str(slots.get("minutes")) + " Minuten"
            if slots.get("seconds"):
                duration_raw = str(slots.get("seconds")) + " Sekunden"

        device_name = slots.get("name")
        device_id = slots.get("device_id")

        # Extract timer description from natural language
        description = await self._extract_description(user_input.text)

        return await self._process_request(
            user_input, duration_raw, device_name, device_id, description
        )

    async def continue_flow(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        step = pending_data.get("step")
        # Restore state
        duration = pending_data.get("duration")
        device_id = pending_data.get("device_id")
        device_name = pending_data.get("name")  # Original query name
        description = pending_data.get("description", "")  # Already extracted description

        text = user_input.text
        learning_data = None

        if step == "ask_duration":
            seconds = parse_duration_string(text)
            if not seconds:
                # If still invalid, ask again
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Ich habe die Zeit nicht verstanden. Bitte sag z.B. '5 Minuten'.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
            duration = seconds  # Now we have duration (in seconds)

        elif step == "ask_device":
            candidates = pending_data.get("candidates", [])
            matched = await self._fuzzy_match_device(text, candidates)
            if not matched:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Das habe ich nicht verstanden. Welches Gerät?", user_input
                    ),
                    "pending_data": pending_data,
                }
            device_id = matched

            # If we had an original name that failed to match automatically, and user now selected one manually -> LEARN IT
            if device_name:
                learning_data = {
                    "type": "entity",
                    "source": device_name,
                    "target": device_id,
                }

        # Recursively call process request to check if we have everything now
        # Pass what we have. If something is still missing, it will ask for the next thing.
        # Use description from pending_data (already extracted in first turn)
        res = await self._process_request(
            user_input,
            duration,
            device_name=device_name,
            device_id=device_id,
            description=description,
        )

        # Inject learning data if we just resolved the device
        if learning_data:
            res["learning_data"] = learning_data

        return res

    async def _extract_description(self, text: str) -> str:
        """Extract timer description using LLM."""
        try:
            result = await self._safe_prompt(
                self.PROMPT, {"user_input": text}, temperature=0.0
            )
            if result and isinstance(result, dict):
                desc = result.get("description", "").strip()
                # Limit to 30 characters for Android timer label
                return desc[:30] if desc else ""
        except Exception as e:
            _LOGGER.debug(f"Failed to extract timer description: {e}")
        return ""

    async def _process_request(
        self, user_input, duration_raw, device_name=None, device_id=None, description=""
    ) -> Dict[str, Any]:
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
                "result": await make_response(
                    "Wie lange soll der Timer laufen?", user_input
                ),
                "pending_data": {
                    "type": "timer",
                    "step": "ask_duration",
                    "device_id": device_id,
                    "name": device_name,
                    "description": description,
                },
            }

        # 2. Resolve Device
        if not device_id:
            mobile_services = self._get_mobile_notify_services()
            if not mobile_services:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Keine mobilen Geräte gefunden.", user_input
                    ),
                }

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
                        "result": await make_response(
                            f"Auf welchem Gerät? ({', '.join(names)})", user_input
                        ),
                        "pending_data": {
                            "type": "timer",
                            "step": "ask_device",
                            "duration": seconds,  # Pass resolved duration forward
                            "candidates": mobile_services,
                            "name": device_name,  # Keep original name for learning
                            "description": description,  # Keep description for final call
                        },
                    }

        # 3. Execute
        await self._set_android_timer(device_id, seconds, description)

        # Get friendly name
        device_friendly = (
            device_id.split(".")[-1]
            .replace("mobile_app_", "")
            .replace("_", " ")
            .title()
        )
        services = self._get_mobile_notify_services()
        for s in services:
            if s["service"] == device_id:
                device_friendly = s["name"]
                break

        # Include description in response if present
        response_text = f"Timer für {format_seconds_to_string(seconds)} auf {device_friendly} gestellt."
        if description:
            response_text = f"Timer '{description}' für {format_seconds_to_string(seconds)} auf {device_friendly} gestellt."

        return {
            "status": "handled",
            "result": await make_response(response_text, user_input),
        }

    def _get_mobile_notify_services(self) -> List[Dict[str, str]]:
        services = self.hass.services.async_services().get("notify", {})
        return [
            {
                "service": f"notify.{k}",
                "name": k.replace("mobile_app_", "").replace("_", " ").title(),
            }
            for k in services
            if k.startswith("mobile_app_")
        ]

    async def _fuzzy_match_device(
        self, query: str, candidates: List[Dict[str, str]]
    ) -> Optional[str]:
        if not query:
            return None

        # Extract names and service IDs from candidates for fuzzy matching
        candidate_names = {c["name"]: c["service"] for c in candidates}
        candidate_ids = {c["service"].split(".")[-1]: c["service"] for c in candidates}

        # Try matching by name first
        match_result = await fuzzy_match_best(
            query, list(candidate_names.keys()), threshold=70
        )
        if match_result:
            best_match_name, score = match_result
            _LOGGER.debug(
                "[Timer] Matched device name '%s' to '%s' (score: %d)",
                query,
                best_match_name,
                score,
            )
            return candidate_names[best_match_name]

        # If no name match, try matching by service ID (last part of the entity_id)
        match_result = await fuzzy_match_best(
            query, list(candidate_ids.keys()), threshold=70
        )
        if match_result:
            best_match_id, score = match_result
            _LOGGER.debug(
                "[Timer] Matched device ID '%s' to '%s' (score: %d)",
                query,
                best_match_id,
                score,
            )
            return candidate_ids[best_match_id]

        _LOGGER.warning("[Timer] No device match for '%s'", query)
        return None

    async def _set_android_timer(
        self, service_full: str, seconds: int, description: str = ""
    ):
        domain, service = service_full.split(".", 1)
        # Build intent extras with optional LABEL
        extras = f"android.intent.extra.alarm.LENGTH:{seconds},android.intent.extra.alarm.SKIP_UI:true"
        if description:
            # Add timer label/description
            extras += f",android.intent.extra.alarm.MESSAGE:{description}"

        payload = {
            "message": "command_activity",
            "data": {
                "intent_action": "android.intent.action.SET_TIMER",
                "intent_extras": extras,
            },
        }
        await self.hass.services.async_call(domain, service, payload)
