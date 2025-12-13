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
        if slots.get("location"):
            event_data["location"] = slots["location"]
        if slots.get("calendar"):
            event_data["calendar_id"] = slots["calendar"]
        
        # Handle date/time combination from slots
        slot_date = slots.get("date", "")  # e.g., "morgen"
        slot_time = slots.get("time", "")  # e.g., "15 Uhr" or "15 Uhr bis 18 Uhr"
        slot_duration = slots.get("duration", "")  # e.g., "3 Stunden"
        
        if slot_date and slot_time:
            # Combine date and time
            # Extract start time
            time_match = re.search(r'(\d{1,2})(?:[:\.](\d{2}))?\s*[Uu]hr', slot_time)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                # Store combined datetime (will be resolved later)
                event_data["start_date_time"] = f"{slot_date} {hour:02d}:{minute:02d}"
                
                # Check for end time "bis X Uhr"
                end_match = re.search(r'bis\s+(\d{1,2})(?:[:\.](\d{2}))?\s*[Uu]hr', slot_time)
                if end_match:
                    end_hour = int(end_match.group(1))
                    end_minute = int(end_match.group(2) or 0)
                    event_data["end_date_time"] = f"{slot_date} {end_hour:02d}:{end_minute:02d}"
            else:
                # No time extracted, use date only
                event_data["start_date"] = slot_date
        elif slot_date and not slot_time:
            # Date only, no time - all-day event
            event_data["start_date"] = slot_date
        
        # Parse duration from slots (e.g., "3 Stunden" -> 180 minutes)
        if slot_duration and not event_data.get("duration_minutes"):
            duration_minutes = self._parse_duration(slot_duration)
            if duration_minutes:
                event_data["duration_minutes"] = duration_minutes
            
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
        
        # 4. Resolve relative dates (morgen, übermorgen, heute, etc.)
        event_data = self._resolve_relative_dates(event_data)
        
        # 5. Validate date formats - if still invalid after resolution, ask again
        if not self._validate_dates(event_data):
            _LOGGER.debug("[Calendar] Date validation failed after resolution: %s", event_data)
            return {
                "status": "handled",
                "result": await make_response(
                    "Bitte gib ein konkretes Datum an, z.B. 'am 14. Dezember' oder 'am Montag um 15 Uhr'.",
                    user_input,
                ),
                "pending_data": {
                    "type": "calendar",
                    "step": "ask_datetime",
                    "event_data": {k: v for k, v in event_data.items() 
                                  if k not in ("start_date", "end_date", "start_date_time", "end_date_time")},
                },
            }
        
        # 5. Calculate end time if not specified
        if not event_data.get("end_date") and not event_data.get("end_date_time"):
            if event_data.get("start_date_time"):
                # Timed event - add duration (default 1 hour)
                duration = event_data.get("duration_minutes", 60)
                try:
                    start = datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
                    end = start + timedelta(minutes=duration)
                    event_data["end_date_time"] = end.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass  # Will be caught by validation
            elif event_data.get("start_date"):
                # All-day event - end date is day after
                try:
                    start = datetime.strptime(event_data["start_date"], "%Y-%m-%d")
                    end = start + timedelta(days=1)
                    event_data["end_date"] = end.strftime("%Y-%m-%d")
                except ValueError:
                    pass  # Will be caught by validation
        
        # 6. Show confirmation before creating
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
    
    def _resolve_relative_dates(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative date terms to actual dates.
        
        Converts German terms like 'morgen', 'übermorgen', 'heute', weekdays etc.
        to actual YYYY-MM-DD format.
        """
        today = datetime.now()
        
        # Relative date mappings (days from today)
        # IMPORTANT: Order by length (longest first) to avoid "morgen" matching "übermorgen"
        relative_days = [
            ("in drei tagen", 3),
            ("in 3 tagen", 3),
            ("übermorgen", 2),
            ("morgen", 1),
            ("heute", 0),
        ]
        
        # German weekday names to weekday number (0=Monday, 6=Sunday)
        weekdays = {
            "montag": 0,
            "dienstag": 1,
            "mittwoch": 2,
            "donnerstag": 3,
            "freitag": 4,
            "samstag": 5,
            "sonntag": 6,
        }
        
        def resolve_date(value: str) -> Optional[str]:
            """Resolve a single date value to YYYY-MM-DD format."""
            if not value or not isinstance(value, str):
                return value
            
            val_lower = value.lower().strip()
            
            # Already in correct format?
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return value  # Already valid
            except ValueError:
                pass
            
            # Check relative days (list of tuples, longest first)
            for term, days in relative_days:
                if term in val_lower:
                    target = today + timedelta(days=days)
                    return target.strftime("%Y-%m-%d")
            
            # Check weekdays (next occurrence)
            for day_name, day_num in weekdays.items():
                if day_name in val_lower:
                    # Find next occurrence of this weekday
                    days_ahead = day_num - today.weekday()
                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7
                    target = today + timedelta(days=days_ahead)
                    return target.strftime("%Y-%m-%d")
            
            # Check for "in einer woche" patterns
            if "in einer woche" in val_lower or "heute in einer woche" in val_lower:
                target = today + timedelta(days=7)
                return target.strftime("%Y-%m-%d")
            
            # Couldn't resolve - return as-is
            return value
        
        def resolve_datetime(value: str) -> Optional[str]:
            """Resolve a datetime value, preserving time if present."""
            if not value or not isinstance(value, str):
                return value
            
            # Already in correct format?
            try:
                datetime.strptime(value, "%Y-%m-%d %H:%M")
                return value
            except ValueError:
                pass
            
            # Try to extract time from the value
            time_match = re.search(r'(\d{1,2})[:\.](\d{2})', value)
            if time_match:
                hour, minute = time_match.groups()
                time_str = f"{int(hour):02d}:{int(minute):02d}"
            else:
                # Check for "um X Uhr" pattern
                uhr_match = re.search(r'(\d{1,2})\s*uhr', value.lower())
                if uhr_match:
                    hour = int(uhr_match.group(1))
                    time_str = f"{hour:02d}:00"
                else:
                    time_str = None
            
            # Resolve date part
            date_part = resolve_date(value)
            
            if date_part and time_str:
                # Check if date_part is valid YYYY-MM-DD
                try:
                    datetime.strptime(date_part, "%Y-%m-%d")
                    return f"{date_part} {time_str}"
                except ValueError:
                    pass
            
            return value
        
        # Make a copy to avoid modifying original
        result = dict(event_data)
        
        # Resolve each date field
        if result.get("start_date"):
            result["start_date"] = resolve_date(result["start_date"])
        
        if result.get("end_date"):
            result["end_date"] = resolve_date(result["end_date"])
        
        if result.get("start_date_time"):
            result["start_date_time"] = resolve_datetime(result["start_date_time"])
        
        if result.get("end_date_time"):
            result["end_date_time"] = resolve_datetime(result["end_date_time"])
        
        _LOGGER.debug("[Calendar] Resolved dates: %s -> %s", event_data, result)
        return result
    
    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string to minutes.
        
        Examples:
            "3 Stunden" -> 180
            "30 Minuten" -> 30
            "1,5 Stunden" -> 90
            "2 Stunden 30 Minuten" -> 150
        """
        if not duration_str:
            return None
        
        total_minutes = 0
        text = duration_str.lower()
        
        # Match hours (X Stunden, X,5 Stunden, etc.)
        hours_match = re.search(r'(\d+(?:[,\.]\d+)?)\s*(?:stunde|stunden|std|h)', text)
        if hours_match:
            hours = float(hours_match.group(1).replace(',', '.'))
            total_minutes += int(hours * 60)
        
        # Match minutes
        minutes_match = re.search(r'(\d+)\s*(?:minute|minuten|min|m)', text)
        if minutes_match:
            total_minutes += int(minutes_match.group(1))
        
        return total_minutes if total_minutes > 0 else None
    
    def _validate_dates(self, event_data: Dict[str, Any]) -> bool:
        """Validate that date fields are in parseable format.
        
        Returns True if dates are valid or not present.
        Returns False if dates are present but unparseable.
        """
        # Check start_date_time format (YYYY-MM-DD HH:MM)
        if event_data.get("start_date_time"):
            try:
                datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
            except ValueError:
                _LOGGER.debug(
                    "[Calendar] Invalid start_date_time format: %s",
                    event_data["start_date_time"]
                )
                return False
        
        # Check start_date format (YYYY-MM-DD)
        if event_data.get("start_date"):
            try:
                datetime.strptime(event_data["start_date"], "%Y-%m-%d")
            except ValueError:
                _LOGGER.debug(
                    "[Calendar] Invalid start_date format: %s",
                    event_data["start_date"]
                )
                return False
        
        # Check end_date_time format
        if event_data.get("end_date_time"):
            try:
                datetime.strptime(event_data["end_date_time"], "%Y-%m-%d %H:%M")
            except ValueError:
                _LOGGER.debug(
                    "[Calendar] Invalid end_date_time format: %s",
                    event_data["end_date_time"]
                )
                return False
        
        # Check end_date format
        if event_data.get("end_date"):
            try:
                datetime.strptime(event_data["end_date"], "%Y-%m-%d")
            except ValueError:
                _LOGGER.debug(
                    "[Calendar] Invalid end_date format: %s",
                    event_data["end_date"]
                )
                return False
        
        return True
    
    def _build_confirmation_text(self, event_data: Dict[str, Any]) -> str:
        """Build a human-readable confirmation text."""
        lines = []
        lines.append(f"📅 **{event_data.get('summary', 'Termin')}**")
        
        if event_data.get("start_date_time"):
            try:
                dt = datetime.strptime(event_data["start_date_time"], "%Y-%m-%d %H:%M")
                lines.append(f"🕐 {dt.strftime('%d.%m.%Y um %H:%M Uhr')}")
                if event_data.get("end_date_time"):
                    end_dt = datetime.strptime(event_data["end_date_time"], "%Y-%m-%d %H:%M")
                    lines.append(f"   bis {end_dt.strftime('%H:%M Uhr')}")
            except ValueError:
                lines.append(f"🕐 {event_data['start_date_time']}")
        elif event_data.get("start_date"):
            try:
                dt = datetime.strptime(event_data["start_date"], "%Y-%m-%d")
                lines.append(f"📆 {dt.strftime('%d.%m.%Y')} (ganztägig)")
            except ValueError:
                lines.append(f"📆 {event_data['start_date']} (ganztägig)")
            
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
