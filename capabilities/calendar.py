"""Calendar capability for creating calendar events via Home Assistant."""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant
from homeassistant.components import conversation

from .base import Capability
from custom_components.multistage_assist.conversation_utils import make_response
from ..utils.fuzzy_utils import fuzzy_match_best


_LOGGER = logging.getLogger(__name__)


class CalendarCapability(Capability):
    """Create calendar events on Home Assistant calendars."""
    
    name = "calendar"
    description = "Create calendar events on connected calendars."
    
    # Prompt to extract calendar event details from natural language
    PROMPT = {
        "system": """Extract calendar event details from the user's request.

Parse the following information if present:
- summary: Event title/name (required)
- description: Additional details about the event
- start_date: Start date in YYYY-MM-DD format (for all-day events)
- end_date: End date in YYYY-MM-DD format (for all-day events, day AFTER the event ends)
- start_date_time: Start date and time in YYYY-MM-DD HH:MM format (for timed events)
- end_date_time: End date and time in YYYY-MM-DD HH:MM format (for timed events)
- location: Event location
- duration_minutes: Duration in minutes if no end time is specified
- is_all_day: true if no specific time is mentioned

Today's date for reference: {today}

Examples:
"Termin morgen um 10 Uhr beim Zahnarzt" → {"summary": "Zahnarzt", "start_date_time": "2023-12-14 10:00", "duration_minutes": 60}
"Geburtstag am 25. Dezember ganztägig" → {"summary": "Geburtstag", "start_date": "2023-12-25", "end_date": "2023-12-26", "is_all_day": true}
"Meeting in 2 Stunden" → {"summary": "Meeting", "start_date_time": "2023-12-13 14:00", "duration_minutes": 60}
"Arzttermin nächsten Montag 14:30 in der Praxis Dr. Müller" → {"summary": "Arzttermin", "start_date_time": "2023-12-18 14:30", "location": "Praxis Dr. Müller", "duration_minutes": 60}

Output format: JSON object with the extracted fields. Only include fields that are clearly specified.
""",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "start_date_time": {"type": "string"},
                "end_date_time": {"type": "string"},
                "location": {"type": "string"},
                "duration_minutes": {"type": "integer"},
                "is_all_day": {"type": "boolean"},
            },
        },
    }
    
    async def run(
        self, user_input, intent_name: str = None, slots: Dict[str, Any] = None, **_: Any
    ) -> Dict[str, Any]:
        """Handle calendar intent from stage1."""
        slots = slots or {}
        
        # Accept calendar-related intents
        if intent_name and intent_name not in ("HassCalendarCreate", "HassCreateEvent", "HassCalendarAdd"):
            return {}
        
        # Extract event details from natural language using LLM
        event_data = await self._extract_event_details(user_input.text)
        
        if not event_data:
            event_data = {}
        
        # Merge with any slots from NLU
        if slots.get("summary"):
            event_data["summary"] = slots["summary"]
        if slots.get("date"):
            event_data["start_date"] = slots["date"]
        if slots.get("time"):
            event_data["start_time"] = slots["time"]
        if slots.get("location"):
            event_data["location"] = slots["location"]
        if slots.get("calendar"):
            event_data["calendar_id"] = slots["calendar"]
            
        return await self._process_request(user_input, event_data)
    
    async def continue_flow(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Continue multi-turn calendar event creation flow."""
        step = pending_data.get("step")
        event_data = pending_data.get("event_data", {})
        text = user_input.text.strip()
        
        if step == "ask_summary":
            # User is providing the event title
            event_data["summary"] = text
            
        elif step == "ask_datetime":
            # User is providing date/time
            parsed = await self._parse_datetime(text)
            if parsed:
                event_data.update(parsed)
            else:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Ich habe das Datum nicht verstanden. Bitte sag z.B. 'morgen um 10 Uhr' oder '25. Dezember'.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
                
        elif step == "ask_calendar":
            # User selected a calendar
            calendars = pending_data.get("calendars", [])
            matched = await self._fuzzy_match_calendar(text, calendars)
            if not matched:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Das habe ich nicht verstanden. Welcher Kalender?",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
            event_data["calendar_id"] = matched
            
        elif step == "confirm":
            # User is confirming or canceling
            text_lower = text.lower()
            if any(word in text_lower for word in ["ja", "ok", "genau", "richtig", "stimmt", "passt"]):
                # Confirmed - create the event
                return await self._create_event(user_input, event_data)
            elif any(word in text_lower for word in ["nein", "abbrechen", "stop", "cancel"]):
                return {
                    "status": "handled",
                    "result": await make_response("Termin wurde nicht erstellt.", user_input),
                }
            else:
                # Unclear response
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Sag 'Ja' zum Bestätigen oder 'Nein' zum Abbrechen.",
                        user_input,
                    ),
                    "pending_data": pending_data,
                }
        
        # Continue processing with updated data
        return await self._process_request(user_input, event_data)
    
    async def _extract_event_details(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract event details using LLM."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = dict(self.PROMPT)
            prompt["system"] = prompt["system"].format(today=today)
            
            result = await self._safe_prompt(
                prompt, {"user_input": text}, temperature=0.0
            )
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            _LOGGER.debug(f"Failed to extract event details: {e}")
        return None
    
    async def _parse_datetime(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse date/time from user input using LLM."""
        return await self._extract_event_details(f"Termin {text}")
    
    async def _process_request(
        self, user_input, event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the calendar event request, asking for missing information."""
        
        # 1. Check for event title/summary (REQUIRED)
        if not event_data.get("summary"):
            return {
                "status": "handled",
                "result": await make_response(
                    "Wie soll der Termin heißen?", user_input
                ),
                "pending_data": {
                    "type": "calendar",
                    "step": "ask_summary",
                    "event_data": event_data,
                },
            }
        
        # 2. Check for date/time (REQUIRED)
        has_datetime = (
            event_data.get("start_date") or 
            event_data.get("start_date_time")
        )
        if not has_datetime:
            return {
                "status": "handled",
                "result": await make_response(
                    "Wann soll der Termin sein?", user_input
                ),
                "pending_data": {
                    "type": "calendar",
                    "step": "ask_datetime",
                    "event_data": event_data,
                },
            }
        
        # 3. Check for calendar (REQUIRED if multiple calendars exist)
        if not event_data.get("calendar_id"):
            calendars = self._get_calendar_entities()
            if not calendars:
                return {
                    "status": "handled",
                    "result": await make_response(
                        "Keine Kalender gefunden. Bitte richte zuerst einen Kalender in Home Assistant ein.",
                        user_input,
                    ),
                }
            
            if len(calendars) == 1:
                # Auto-select the only calendar
                event_data["calendar_id"] = calendars[0]["entity_id"]
            else:
                # Ask which calendar to use
                calendar_names = [c["name"] for c in calendars]
                return {
                    "status": "handled",
                    "result": await make_response(
                        f"In welchen Kalender? ({', '.join(calendar_names)})",
                        user_input,
                    ),
                    "pending_data": {
                        "type": "calendar",
                        "step": "ask_calendar",
                        "event_data": event_data,
                        "calendars": calendars,
                    },
                }
        
        # 4. Calculate end time if not specified
        if not event_data.get("end_date") and not event_data.get("end_date_time"):
            if event_data.get("start_date_time"):
                # Timed event - add duration (default 1 hour)
                duration = event_data.get("duration_minutes", 60)
                start = datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
                end = start + timedelta(minutes=duration)
                event_data["end_date_time"] = end.strftime("%Y-%m-%d %H:%M")
            elif event_data.get("start_date"):
                # All-day event - end date is day after
                start = datetime.strptime(event_data["start_date"], "%Y-%m-%d")
                end = start + timedelta(days=1)
                event_data["end_date"] = end.strftime("%Y-%m-%d")
        
        # 5. Show confirmation before creating
        summary = self._build_confirmation_text(event_data)
        return {
            "status": "handled",
            "result": await make_response(
                f"Termin erstellen?\n{summary}\n\nSag 'Ja' zum Bestätigen.",
                user_input,
            ),
            "pending_data": {
                "type": "calendar",
                "step": "confirm",
                "event_data": event_data,
            },
        }
    
    def _build_confirmation_text(self, event_data: Dict[str, Any]) -> str:
        """Build a human-readable confirmation text."""
        lines = []
        lines.append(f"📅 **{event_data.get('summary', 'Termin')}**")
        
        if event_data.get("start_date_time"):
            dt = datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
            lines.append(f"🕐 {dt.strftime('%d.%m.%Y um %H:%M Uhr')}")
            if event_data.get("end_date_time"):
                end_dt = datetime.strptime(event_data["end_date_time"], "%Y-%m-%d %H:%M")
                lines.append(f"   bis {end_dt.strftime('%H:%M Uhr')}")
        elif event_data.get("start_date"):
            dt = datetime.strptime(event_data["start_date"], "%Y-%m-%d")
            lines.append(f"📆 {dt.strftime('%d.%m.%Y')} (ganztägig)")
            
        if event_data.get("location"):
            lines.append(f"📍 {event_data['location']}")
            
        if event_data.get("description"):
            lines.append(f"📝 {event_data['description']}")
            
        # Get calendar friendly name
        calendar_id = event_data.get("calendar_id", "")
        calendar_name = calendar_id.replace("calendar.", "").replace("_", " ").title()
        for cal in self._get_calendar_entities():
            if cal["entity_id"] == calendar_id:
                calendar_name = cal["name"]
                break
        lines.append(f"📁 Kalender: {calendar_name}")
        
        return "\n".join(lines)
    
    def _get_calendar_entities(self) -> List[Dict[str, str]]:
        """Get all calendar entities from Home Assistant."""
        calendars = []
        for entity_id in self.hass.states.async_entity_ids("calendar"):
            state = self.hass.states.get(entity_id)
            if state:
                friendly_name = state.attributes.get("friendly_name", entity_id)
                calendars.append({
                    "entity_id": entity_id,
                    "name": friendly_name,
                })
        return calendars
    
    async def _fuzzy_match_calendar(
        self, query: str, calendars: List[Dict[str, str]]
    ) -> Optional[str]:
        """Match user input to a calendar using fuzzy matching."""
        if not query or not calendars:
            return None
        
        # Try matching by name
        calendar_names = {c["name"]: c["entity_id"] for c in calendars}
        match_result = await fuzzy_match_best(
            query, list(calendar_names.keys()), threshold=60
        )
        if match_result:
            best_match_name, score = match_result
            _LOGGER.debug(
                "[Calendar] Matched calendar '%s' to '%s' (score: %d)",
                query, best_match_name, score
            )
            return calendar_names[best_match_name]
        
        # Try matching by entity_id
        calendar_ids = {c["entity_id"].split(".")[-1]: c["entity_id"] for c in calendars}
        match_result = await fuzzy_match_best(
            query, list(calendar_ids.keys()), threshold=60
        )
        if match_result:
            best_match_id, score = match_result
            return calendar_ids[best_match_id]
        
        return None
    
    async def _create_event(
        self, user_input, event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create the calendar event in Home Assistant."""
        calendar_id = event_data.get("calendar_id")
        if not calendar_id:
            return {
                "status": "handled",
                "result": await make_response(
                    "Fehler: Kein Kalender ausgewählt.", user_input
                ),
            }
        
        # Build service data
        service_data = {
            "summary": event_data.get("summary", "Termin"),
        }
        
        # Add date/time (use either timed or all-day format)
        if event_data.get("start_date_time"):
            service_data["start_date_time"] = event_data["start_date_time"] + ":00"
            service_data["end_date_time"] = event_data.get("end_date_time", event_data["start_date_time"]) + ":00"
        elif event_data.get("start_date"):
            service_data["start_date"] = event_data["start_date"]
            service_data["end_date"] = event_data.get("end_date", event_data["start_date"])
        
        # Add optional fields
        if event_data.get("description"):
            service_data["description"] = event_data["description"]
        if event_data.get("location"):
            service_data["location"] = event_data["location"]
        
        try:
            await self.hass.services.async_call(
                "calendar",
                "create_event",
                service_data,
                target={"entity_id": calendar_id},
            )
            
            summary = event_data.get("summary", "Termin")
            return {
                "status": "handled",
                "result": await make_response(
                    f"✅ Termin '{summary}' wurde erstellt.", user_input
                ),
            }
        except Exception as e:
            _LOGGER.error(f"Failed to create calendar event: {e}")
            return {
                "status": "handled",
                "result": await make_response(
                    f"Fehler beim Erstellen des Termins: {str(e)}", user_input
                ),
            }
