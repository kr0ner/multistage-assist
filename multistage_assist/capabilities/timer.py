"""Timer capability using MultiTurnCapability base class."""

import logging
from typing import Any, Dict, List, Optional

from .multi_turn_base import MultiTurnCapability
from custom_components.multistage_assist.conversation_utils import (
    make_response,
    parse_duration_string,
    format_seconds_to_string,
)
from ..utils.fuzzy_utils import fuzzy_match_candidates


_LOGGER = logging.getLogger(__name__)


class TimerCapability(MultiTurnCapability):
    """Set timers on mobile devices."""
    
    name = "timer"
    description = "Set timers on mobile devices."
    
    # Field definitions
    MANDATORY_FIELDS = ["duration", "device_id"]
    OPTIONAL_FIELDS = ["description"]
    
    FIELD_PROMPTS = {
        "duration": "Wie lange soll der Timer laufen?",
        "device_id": "Auf welchem Gerät?",  # Will be customized with device list
    }
    
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
    
    # Store mobile services list for device selection
    _mobile_services: List[Dict[str, str]] = []
    _original_device_name: Optional[str] = None  # For learning data
    
    async def run(
        self, user_input, intent_name: str = None, slots: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Handle timer intent."""
        slots = slots or {}
        
        # Accept both intents
        if intent_name and intent_name not in ("HassTimerSet", "HassStartTimer"):
            return {}
        
        # Pre-process: Parse duration from slots
        duration_raw = slots.get("duration")
        if not duration_raw:
            if slots.get("minutes"):
                duration_raw = str(slots.get("minutes")) + " Minuten"
            if slots.get("seconds"):
                duration_raw = str(slots.get("seconds")) + " Sekunden"
        
        if duration_raw:
            seconds = parse_duration_string(duration_raw) if isinstance(duration_raw, str) else duration_raw
            slots["duration"] = seconds
        
        # Store device name for learning
        self._original_device_name = slots.get("name")
        
        return await super().run(user_input, intent_name, slots, **kwargs)
    
    async def _extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract timer fields from natural language."""
        result = {}
        
        # Try to extract duration from text
        duration = parse_duration_string(text)
        if duration:
            result["duration"] = duration
        
        # Try to extract description via LLM
        desc = await self._extract_description(text)
        if desc:
            result["description"] = desc
        
        return result
    
    async def _extract_description(self, text: str) -> str:
        """Extract timer description using LLM."""
        try:
            result = await self._safe_prompt(
                self.PROMPT, {"user_input": text}, temperature=0.0
            )
            if result and isinstance(result, dict):
                desc = result.get("description", "").strip()
                return desc[:30] if desc else ""  # Limit for Android
        except Exception as e:
            _LOGGER.debug(f"Failed to extract timer description: {e}")
        return ""
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and auto-resolve device if possible."""
        # Refresh mobile services list
        self._mobile_services = self._get_mobile_notify_services()
        
        if not self._mobile_services:
            # No devices available - this will fail later
            return data
        
        # If no device_id yet, try to auto-select or fuzzy match
        if not data.get("device_id"):
            # Try fuzzy match on original name
            if self._original_device_name:
                matched = await self._fuzzy_match_device(
                    self._original_device_name, self._mobile_services
                )
                if matched:
                    data["device_id"] = matched
            
            # Auto-select if only one device
            if not data.get("device_id") and len(self._mobile_services) == 1:
                data["device_id"] = self._mobile_services[0]["service"]
                _LOGGER.debug("[Timer] Auto-selected single device: %s", data["device_id"])
        
        return data
    
    async def _ask_for_field(
        self, user_input, field: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Custom field prompts for timer."""
        if field == "device_id":
            # Check if no devices available
            if not self._mobile_services:
                return {
                    "status": "handled",
                    "result": await make_response("Keine mobilen Geräte gefunden.", user_input),
                }
            
            # Show available devices
            names = [d["name"] for d in self._mobile_services]
            prompt = f"Auf welchem Gerät? ({', '.join(names)})"
            
            return {
                "status": "handled",
                "result": await make_response(prompt, user_input),
                "pending_data": {
                    "type": self.name,
                    "step": field,
                    "data": data,
                    "candidates": self._mobile_services,
                    "original_name": self._original_device_name,
                },
            }
        
        # Default behavior for other fields
        return await super()._ask_for_field(user_input, field, data)
    
    async def _parse_field_value(self, field: str, text: str) -> Any:
        """Parse user's response for a field."""
        if field == "duration":
            return parse_duration_string(text)
        return text
    
    async def continue_flow(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Continue multi-turn flow - handles device matching."""
        step = pending_data.get("step")
        data = pending_data.get("data", {})
        text = user_input.text.strip()
        
        # Restore state
        self._original_device_name = pending_data.get("original_name")
        candidates = pending_data.get("candidates", [])
        if candidates:
            self._mobile_services = candidates
        
        learning_data = None
        
        if step == "duration":
            seconds = parse_duration_string(text)
            if not seconds:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Ich habe die Zeit nicht verstanden. Bitte sag z.B. '5 Minuten'.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
            data["duration"] = seconds
            
        elif step == "device_id":
            matched = await self._fuzzy_match_device(text, self._mobile_services)
            if not matched:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Das habe ich nicht verstanden. Welches Gerät?", user_input
                    ),
                    "pending_data": pending_data,
                }
            data["device_id"] = matched
            
            # Learn device alias if we had an original name that failed
            if self._original_device_name:
                learning_data = {
                    "type": "entity",
                    "source": self._original_device_name,
                    "target": matched,
                }
        
        elif step == "confirm":
            # Handle confirmation via parent
            return await super().continue_flow(user_input, pending_data)
        
        # Continue processing
        result = await self._process(user_input, data)
        
        # Inject learning data if applicable
        if learning_data and "pending_data" not in result:
            result["learning_data"] = learning_data
        
        return result
    
    def _needs_confirmation(self) -> bool:
        """Timer doesn't need confirmation - execute directly."""
        return False
    
    async def _build_confirmation(self, data: Dict[str, Any]) -> str:
        """Build confirmation text (not used since confirmation disabled)."""
        duration = data.get("duration", 0)
        device_id = data.get("device_id", "")
        description = data.get("description", "")
        
        device_friendly = self._get_device_friendly_name(device_id)
        duration_str = format_seconds_to_string(duration)
        
        if description:
            return f"Timer '{description}' für {duration_str} auf {device_friendly}?"
        return f"Timer für {duration_str} auf {device_friendly}?"
    
    async def _execute(self, user_input, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the timer."""
        duration = data.get("duration", 0)
        device_id = data.get("device_id", "")
        description = data.get("description", "")
        
        # Set the timer
        await self._set_android_timer(device_id, duration, description)
        
        # Build response
        device_friendly = self._get_device_friendly_name(device_id)
        duration_str = format_seconds_to_string(duration)
        
        if description:
            response = f"Timer '{description}' für {duration_str} auf {device_friendly} gestellt."
        else:
            response = f"Timer für {duration_str} auf {device_friendly} gestellt."
        
        return {
            "status": "handled",
            "result": await make_response(response, user_input),
        }
    
    # --- Helper methods ---
    
    def _get_mobile_notify_services(self) -> List[Dict[str, str]]:
        """Get list of mobile app notification services."""
        from ..utils.service_discovery import get_mobile_notify_services
        return get_mobile_notify_services(self.hass)
    
    def _get_device_friendly_name(self, device_id: str) -> str:
        """Get friendly name for a device."""
        for s in self._mobile_services:
            if s["service"] == device_id:
                return s["name"]
        # Fallback
        return (
            device_id.split(".")[-1]
            .replace("mobile_app_", "")
            .replace("_", " ")
            .title()
        )
    
    async def _fuzzy_match_device(
        self, query: str, candidates: List[Dict[str, str]]
    ) -> Optional[str]:
        """Fuzzy match device from query."""
        # Use centralized fuzzy matching utility
        # Timer uses "service" instead of "entity_id" for the ID key
        return await fuzzy_match_candidates(
            query,
            candidates,
            name_key="name",
            id_key="service",
            threshold=70,
        )
    
    async def _set_android_timer(
        self, service_full: str, seconds: int, description: str = ""
    ):
        """Set Android timer via notification service."""
        domain, service = service_full.split(".", 1)
        
        extras = f"android.intent.extra.alarm.LENGTH:{seconds},android.intent.extra.alarm.SKIP_UI:true"
        if description:
            extras += f",android.intent.extra.alarm.MESSAGE:{description}"
        
        payload = {
            "message": "command_activity",
            "data": {
                "intent_action": "android.intent.action.SET_TIMER",
                "intent_extras": extras,
            },
        }
        await self.hass.services.async_call(domain, service, payload)
