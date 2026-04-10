"""Service discovery utilities for Home Assistant entities and services.

Provides centralized logic for discovering entities, services, and their
attributes from Home Assistant.
"""

import logging
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


# --- Entity Discovery ---

def get_entities_by_domain(
    hass: HomeAssistant,
    domain: str,
    check_exposure: bool = True,
) -> List[Dict[str, Any]]:
    """Get all entities for a domain with their attributes.
    
    Args:
        hass: Home Assistant instance
        domain: Entity domain (e.g., "light", "cover", "calendar")
        check_exposure: Whether to filter by conversation exposure
        
    Returns:
        List of dicts with entity_id, friendly_name, and state
        
    Example:
        calendars = get_entities_by_domain(hass, "calendar")
        # [{"entity_id": "calendar.family", "name": "Family", "state": "on"}, ...]
    """
    entities = []
    
    # Get entity IDs for domain
    try:
        entity_ids = hass.states.async_entity_ids(domain)
    except Exception:
        # Fallback for tests or unusual setups
        entity_ids = []
        try:
            for state in hass.states.async_all(domain):
                entity_ids.append(state.entity_id)
        except Exception:
            _LOGGER.debug("[ServiceDiscovery] Could not get entities for domain %s", domain)
            return []
    
    # Check exposure if requested
    if check_exposure:
        try:
            from homeassistant.components.conversation import async_should_expose
            use_exposure = True
        except ImportError:
            use_exposure = False
    else:
        use_exposure = False
    
    for entity_id in entity_ids:
        # Filter by exposure
        if use_exposure:
            try:
                exposed = async_should_expose(hass, "conversation", entity_id)
                if not exposed:
                    _LOGGER.debug(
                        "[ServiceDiscovery] Skipping %s (not exposed to conversation)",
                        entity_id
                    )
                    continue
            except Exception as e:
                _LOGGER.debug(
                    "[ServiceDiscovery] Exposure check failed for %s: %s, including anyway",
                    entity_id, e
                )
        
        # Get state and attributes
        state = hass.states.get(entity_id)
        if state:
            name = state.attributes.get("friendly_name", entity_id.split(".")[-1])
            entities.append({
                "entity_id": entity_id,
                "name": name,
                "state": state.state,
                "attributes": dict(state.attributes),
            })
    
    _LOGGER.debug(
        "[ServiceDiscovery] Found %d %s entities (check_exposure=%s)",
        len(entities), domain, check_exposure
    )
    return entities


def get_services_by_domain(
    hass: HomeAssistant,
    domain: str,
) -> List[str]:
    """Get all service names for a domain.
    
    Args:
        hass: Home Assistant instance
        domain: Service domain (e.g., "notify", "light")
        
    Returns:
        List of service names (without domain prefix)
        
    Example:
        services = get_services_by_domain(hass, "notify")
        # ["mobile_app_phone", "persistent_notification", ...]
    """
    try:
        all_services = hass.services.async_services()
        domain_services = all_services.get(domain, {})
        return list(domain_services.keys())
    except Exception as e:
        _LOGGER.debug("[ServiceDiscovery] Could not get services for %s: %s", domain, e)
        return []


def get_services_matching(
    hass: HomeAssistant,
    domain: str,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Get services matching a pattern with formatted names.
    
    Args:
        hass: Home Assistant instance
        domain: Service domain
        prefix: Optional prefix filter (e.g., "mobile_app_")
        suffix: Optional suffix filter
        
    Returns:
        List of dicts with service (full name) and name (friendly)
        
    Example:
        mobile = get_services_matching(hass, "notify", prefix="mobile_app_")
        # [{"service": "notify.mobile_app_phone", "name": "Phone"}, ...]
    """
    services = get_services_by_domain(hass, domain)
    result = []
    
    for service_name in services:
        # Apply filters
        if prefix and not service_name.startswith(prefix):
            continue
        if suffix and not service_name.endswith(suffix):
            continue
        
        # Build friendly name
        friendly = service_name
        if prefix:
            friendly = friendly.replace(prefix, "")
        if suffix:
            friendly = friendly.replace(suffix, "")
        friendly = friendly.replace("_", " ").title()
        
        result.append({
            "service": f"{domain}.{service_name}",
            "name": friendly,
        })
    
    return result


def get_mobile_notify_services(hass: HomeAssistant) -> List[Dict[str, str]]:
    """Get mobile app notification services.
    
    Shortcut for common pattern of finding mobile_app_* notify services.
    
    Args:
        hass: Home Assistant instance
        
    Returns:
        List of dicts with service and name
        
    Example:
        devices = get_mobile_notify_services(hass)
        # [{"service": "notify.mobile_app_phone", "name": "Phone"}, ...]
    """
    return get_services_matching(hass, "notify", prefix="mobile_app_")
