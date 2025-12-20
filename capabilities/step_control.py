"""Generic Step Control Capability for relative adjustments.

Provides domain-agnostic step up/down functionality for entities that support
relative value adjustments (brightness, position, temperature, speed, etc.).

Supported Domains:
- light: brightness (0-100%)
- cover: position (0-100%)  
- fan: percentage (0-100%)
- climate: temperature (absolute step)

Usage in intent slots:
    command: "step_up" or "step_down"
    
Example commands:
    "Mach das Licht heller" -> step_up on light
    "Rollo halb hoch" -> step_up on cover
    "Heizung wärmer" -> step_up on climate
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from homeassistant.core import HomeAssistant

from .base import Capability
from ..constants.domain_config import get_step_config, DOMAIN_CONFIG

_LOGGER = logging.getLogger(__name__)


class StepControlCapability(Capability):
    """Calculate step values for relative entity adjustments.
    
    This capability is called before intent execution to resolve
    step_up/step_down commands into concrete values.
    """
    
    name = "step_control"
    description = "Calculate step values for relative adjustments (heller/dunkler, wärmer/kälter)."
    
    # Default step configuration (fallback if not in DOMAIN_CONFIG)
    DEFAULT_STEP_PERCENT = 25
    DEFAULT_MIN_STEP = 10
    DEFAULT_OFF_TO_ON = 30
    
    # Climate-specific defaults
    DEFAULT_TEMP_STEP = 1.0
    DEFAULT_MIN_TEMP = 16
    DEFAULT_MAX_TEMP = 28
    
    async def run(
        self,
        user_input,
        *,
        entity_id: str,
        command: str,  # "step_up" or "step_down"
        domain: str = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate the new value for a step command.
        
        Args:
            user_input: Conversation input
            entity_id: Target entity
            command: "step_up" or "step_down"
            domain: Entity domain (auto-detected if not provided)
            
        Returns:
            Dict with:
                - new_value: Calculated target value
                - attribute: Attribute name to set
                - current_value: Current value before step
                - step_applied: Amount of change applied
        """
        if command not in ("step_up", "step_down"):
            return {}
        
        # Auto-detect domain
        if not domain:
            domain = entity_id.split(".", 1)[0] if "." in entity_id else None
        
        if not domain:
            return {}
        
        # Get step configuration for this domain
        step_config = get_step_config(domain) or self._get_default_config(domain)
        
        if not step_config:
            _LOGGER.debug("[StepControl] No step config for domain '%s'", domain)
            return {}
        
        # Get current state
        state = self.hass.states.get(entity_id)
        if not state:
            _LOGGER.warning("[StepControl] Entity %s not found", entity_id)
            return {}
        
        # Calculate new value based on domain type
        attribute = step_config.get("attribute")
        
        if domain == "climate":
            result = self._calculate_climate_step(state, command, step_config)
        else:
            result = self._calculate_percentage_step(state, command, step_config, domain)
        
        if result:
            _LOGGER.debug(
                "[StepControl] %s on %s: %s %s -> %s (change: %s)",
                command, entity_id, attribute,
                result.get("current_value"),
                result.get("new_value"),
                result.get("step_applied"),
            )
        
        return result
    
    def _calculate_percentage_step(
        self,
        state,
        command: str,
        config: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        """Calculate step for percentage-based attributes (light, cover, fan).
        
        Args:
            state: Entity state object
            command: step_up or step_down
            config: Step configuration
            domain: Entity domain
            
        Returns:
            Dict with new_value, attribute, current_value, step_applied
        """
        attribute = config.get("attribute", "brightness")
        step_percent = config.get("step_percent", self.DEFAULT_STEP_PERCENT)
        min_step = config.get("min_step", self.DEFAULT_MIN_STEP)
        off_to_on = config.get("off_to_on", self.DEFAULT_OFF_TO_ON)
        
        # Get current value
        if domain == "light":
            # Light brightness is 0-255, convert to percentage
            raw_value = state.attributes.get(attribute, 0) or 0
            current_pct = int((raw_value / 255.0) * 100)
        elif domain == "cover":
            # Cover position is already 0-100 (but 0=closed, 100=open)
            current_pct = state.attributes.get("current_position", 0) or 0
        elif domain == "fan":
            # Fan percentage is already 0-100
            current_pct = state.attributes.get(attribute, 0) or 0
        else:
            # Generic percentage attribute
            current_pct = state.attributes.get(attribute, 0) or 0
        
        # Handle off state
        is_off = state.state in ("off", "closed", "unavailable")
        
        if command == "step_up":
            if is_off or current_pct == 0:
                # Entity is off, turn on to default value
                new_pct = off_to_on
                step_applied = off_to_on
            else:
                # Calculate percentage-based step
                step_applied = max(min_step, int(current_pct * step_percent / 100))
                new_pct = min(100, current_pct + step_applied)
        else:  # step_down
            if is_off or current_pct == 0:
                # Already off, nothing to do
                return {}
            
            step_applied = max(min_step, int(current_pct * step_percent / 100))
            new_pct = max(0, current_pct - step_applied)
        
        return {
            "new_value": new_pct,
            "attribute": attribute,
            "current_value": current_pct,
            "step_applied": step_applied,
            "domain": domain,
        }
    
    def _calculate_climate_step(
        self,
        state,
        command: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate step for climate temperature (absolute step).
        
        Args:
            state: Entity state object
            command: step_up or step_down
            config: Step configuration
            
        Returns:
            Dict with new_value, attribute, current_value, step_applied
        """
        attribute = config.get("attribute", "temperature")
        step_absolute = config.get("step_absolute", self.DEFAULT_TEMP_STEP)
        min_temp = config.get("min_temp", self.DEFAULT_MIN_TEMP)
        max_temp = config.get("max_temp", self.DEFAULT_MAX_TEMP)
        
        # Get current temperature
        current_temp = state.attributes.get("temperature")
        if current_temp is None:
            current_temp = state.attributes.get("current_temperature")
        
        if current_temp is None:
            _LOGGER.debug("[StepControl] No temperature attribute found for climate entity")
            return {}
        
        current_temp = float(current_temp)
        
        if command == "step_up":
            new_temp = min(max_temp, current_temp + step_absolute)
        else:  # step_down
            new_temp = max(min_temp, current_temp - step_absolute)
        
        return {
            "new_value": new_temp,
            "attribute": attribute,
            "current_value": current_temp,
            "step_applied": step_absolute,
            "domain": "climate",
        }
    
    def _get_default_config(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get default step config for a domain not in DOMAIN_CONFIG."""
        defaults = {
            "light": {
                "attribute": "brightness",
                "step_percent": 35,
                "min_step": 10,
                "off_to_on": 30,
            },
            "cover": {
                "attribute": "position",
                "step_percent": 25,
                "min_step": 10,
                "off_to_on": 100,
            },
            "fan": {
                "attribute": "percentage",
                "step_percent": 25,
                "min_step": 10,
                "off_to_on": 50,
            },
            "climate": {
                "attribute": "temperature",
                "step_absolute": 1.0,
                "min_temp": 16,
                "max_temp": 28,
            },
        }
        return defaults.get(domain)
    
    async def apply_step(
        self,
        entity_id: str,
        command: str,
        domain: str = None,
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Convenience method to calculate and return attribute/value pair.
        
        Args:
            entity_id: Target entity
            command: step_up or step_down
            domain: Optional domain
            
        Returns:
            Tuple of (attribute_name, new_value) or (None, None) if failed
        """
        result = await self.run(None, entity_id=entity_id, command=command, domain=domain)
        
        if not result:
            return None, None
        
        return result.get("attribute"), result.get("new_value")


# Helper function for use outside capability context
def calculate_step(
    hass: HomeAssistant,
    entity_id: str,
    command: str,
) -> Dict[str, Any]:
    """Calculate step value for an entity.
    
    Args:
        hass: Home Assistant instance
        entity_id: Target entity
        command: step_up or step_down
        
    Returns:
        Dict with new_value, attribute, current_value, step_applied
    """
    cap = StepControlCapability(hass, {})
    # Note: This is synchronous access to state, used for quick calculations
    
    domain = entity_id.split(".", 1)[0]
    state = hass.states.get(entity_id)
    
    if not state:
        return {}
    
    step_config = get_step_config(domain) or cap._get_default_config(domain)
    
    if not step_config:
        return {}
    
    if domain == "climate":
        return cap._calculate_climate_step(state, command, step_config)
    else:
        return cap._calculate_percentage_step(state, command, step_config, domain)
