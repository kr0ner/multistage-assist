"""Base class for multi-turn conversation capabilities."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base import Capability
from custom_components.multistage_assist.conversation_utils import make_response

_LOGGER = logging.getLogger(__name__)


class MultiTurnCapability(Capability):
    """
    Base class for capabilities that need multi-turn conversations.
    
    Subclasses define mandatory and optional fields. The base class handles:
    - Prompting for missing mandatory fields
    - Extracting optional fields without prompting
    - Confirmation flow
    - Pending state management
    
    Subclass must define:
    - MANDATORY_FIELDS: List of required field names
    - OPTIONAL_FIELDS: List of optional field names  
    - FIELD_PROMPTS: Dict of field -> prompt text
    - _extract_fields(): Extract all fields from natural language
    - _execute(): Execute the action after all fields are collected
    """
    
    # Subclass defines these
    MANDATORY_FIELDS: List[str] = []
    OPTIONAL_FIELDS: List[str] = []
    
    # What to ask when field is missing
    FIELD_PROMPTS: Dict[str, str] = {}
    
    # Affirmative/negative responses for confirmation
    AFFIRMATIVE = {"ja", "ok", "genau", "richtig", "passt", "korrekt", "stimmt", "gut", "jawohl"}
    NEGATIVE = {"nein", "nicht", "abbrechen", "stop", "stopp", "falsch", "cancel", "weg"}
    
    async def run(
        self, user_input, intent_name: str = None, slots: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Handle initial request - extract fields and process.
        """
        slots = slots or {}
        
        # 1. Extract all fields from natural language
        extracted = await self._extract_fields(user_input.text)
        if not extracted:
            extracted = {}
        
        # 2. Merge with any slots from NLU (slots take precedence)
        data = {**extracted}
        for key, value in slots.items():
            if value:  # Only override if slot has value
                data[key] = value
        
        _LOGGER.debug("[%s] Extracted data: %s", self.name, data)
        
        # 3. Process - will ask for missing mandatory fields
        return await self._process(user_input, data)
    
    async def continue_flow(
        self, user_input, pending_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Continue multi-turn flow with user's response.
        """
        step = pending_data.get("step")
        data = pending_data.get("data", {})
        
        text = user_input.text.strip()
        
        # Handle confirmation step
        if step == "confirm":
            if self._is_affirmative(text):
                return await self._execute(user_input, data)
            elif self._is_negative(text):
                return {
                    "status": "handled",
                    "result": await make_response("Abgebrochen.", user_input),
                }
            else:
                # Unclear response, ask again
                confirmation_text = await self._build_confirmation(data)
                return {
                    "status": "handled",
                    "result": await make_response(
                        f"{confirmation_text}\n\nSag 'Ja' zum Bestätigen oder 'Nein' zum Abbrechen.",
                        user_input
                    ),
                    "pending_data": {"type": self.name, "step": "confirm", "data": data},
                }
        
        # Handle field collection step
        if step in self.MANDATORY_FIELDS:
            # User provided value for this field
            data[step] = await self._parse_field_value(step, text)
            _LOGGER.debug("[%s] Field '%s' set to: %s", self.name, step, data[step])
        
        # Continue processing
        return await self._process(user_input, data)
    
    async def _process(
        self, user_input, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process current state - check for missing fields and proceed.
        """
        # 1. Validate/transform data if needed
        data = await self._validate_data(data)
        
        # 2. Find first missing mandatory field
        for field in self.MANDATORY_FIELDS:
            if not self._has_field(data, field):
                return await self._ask_for_field(user_input, field, data)
        
        # 3. Try to fill optional fields (no prompting)
        for field in self.OPTIONAL_FIELDS:
            if not self._has_field(data, field):
                extracted = await self._extract_single_field(user_input.text, field)
                if extracted:
                    data[field] = extracted
        
        # 4. All mandatory fields present - show confirmation
        if self._needs_confirmation():
            return await self._show_confirmation(user_input, data)
        
        # 5. No confirmation needed - execute directly
        return await self._execute(user_input, data)
    
    def _has_field(self, data: Dict[str, Any], field: str) -> bool:
        """Check if field has a valid value."""
        value = data.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True
    
    async def _ask_for_field(
        self, user_input, field: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask user for a missing field."""
        prompt = self.FIELD_PROMPTS.get(field, f"Bitte gib {field} an.")
        
        return {
            "status": "handled",
            "result": await make_response(prompt, user_input),
            "pending_data": {
                "type": self.name,
                "step": field,
                "data": data,
            },
        }
    
    async def _show_confirmation(
        self, user_input, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Show confirmation before executing."""
        confirmation_text = await self._build_confirmation(data)
        
        return {
            "status": "handled",
            "result": await make_response(
                f"{confirmation_text}\n\nSag 'Ja' zum Bestätigen.",
                user_input
            ),
            "pending_data": {
                "type": self.name,
                "step": "confirm",
                "data": data,
            },
        }
    
    def _is_affirmative(self, text: str) -> bool:
        """Check if text is an affirmative response."""
        from ..utils.german_utils import is_affirmative
        return is_affirmative(text)
    
    def _is_negative(self, text: str) -> bool:
        """Check if text is a negative response."""
        from ..utils.german_utils import is_negative
        return is_negative(text)
    
    # --- Subclass must implement these ---
    
    @abstractmethod
    async def _extract_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract all possible fields from natural language text.
        
        Returns dict of field_name -> value for all fields that could be extracted.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def _execute(self, user_input, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action after all mandatory fields are collected.
        
        Returns the final conversation result.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def _build_confirmation(self, data: Dict[str, Any]) -> str:
        """
        Build confirmation text showing what will be done.
        """
        raise NotImplementedError
    
    # --- Optional overrides ---
    
    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and transform data. Override to add custom validation.
        
        Called before checking for missing fields.
        """
        return data
    
    async def _parse_field_value(self, field: str, text: str) -> Any:
        """
        Parse user's response for a field. Override for custom parsing.
        
        Default: return text as-is.
        """
        return text
    
    async def _extract_single_field(self, text: str, field: str) -> Any:
        """
        Try to extract a single optional field from text.
        
        Override to add field-specific extraction.
        Default: return None (don't extract).
        """
        return None
    
    def _needs_confirmation(self) -> bool:
        """
        Whether to show confirmation before executing.
        
        Override to disable confirmation for simple actions.
        Default: True
        """
        return True
