import logging
from typing import Any, Dict, List, Set, Optional

from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er
from homeassistant.components import conversation

from .base import Capability

_LOGGER = logging.getLogger(__name__)


class EntityResolverCapability(Capability):
    """Resolve entity_ids from simple slot-like inputs (area, domain, name)."""

    name = "entity_resolver"
    description = "Resolves entities by area/domain/friendly name using HA registries."

    def _normalize(self, value: Any) -> Optional[str]:
        """Accept primitives or hassil MatchEntity; return stripped string (or None)."""
        if value is None:
            return None
        raw = getattr(value, "value", value)
        if raw is None:
            return None
        s = str(raw).strip()
        return s or None

    def _collect_area_entities(self, area_name: Optional[str], domain: Optional[str]) -> List[str]:
        if not area_name:
            return []
        area_reg = ar.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)

        area = None
        area_lc = area_name.lower()
        for a in area_reg.async_list_areas():
            if (a.name or "").strip().lower() == area_lc:
                area = a
                break
        if not area:
            _LOGGER.debug("[EntityResolver] Area '%s' not found", area_name)
            return []

        device_ids: Set[str] = {d.id for d in dev_reg.devices.values() if d.area_id == area.id}

        entity_ids: List[str] = []
        for ent in ent_reg.entities.values():
            in_area = (ent.area_id == area.id) or (ent.device_id in device_ids if ent.device_id else False)
            if not in_area:
                continue
            if domain and ent.domain != domain:
                continue
            entity_ids.append(ent.entity_id)
        return entity_ids

    def _collect_by_name(self, name: Optional[str], domain: Optional[str]) -> List[str]:
        if not name:
            return []
        ent_reg = er.async_get(self.hass)
        needle = name.lower()
        matches: List[str] = []
        for ent in ent_reg.entities.values():
            if domain and ent.domain != domain:
                continue
            friendly = (ent.original_name or ent.name or "").strip().lower()
            if friendly == needle or needle in friendly:
                matches.append(ent.entity_id)
        return matches

    async def run(self, user_input: conversation.ConversationInput, **kwargs: Any) -> Dict[str, Any]:
        """Resolve entities from either a provided 'entities' mapping (hassil slots) or explicit kwargs."""
        slots: Dict[str, Any] = kwargs.get("entities") or {}
        if not isinstance(slots, dict):
            slots = {}

        area_name = self._normalize(slots.get("area") or kwargs.get("area"))
        domain = self._normalize(slots.get("domain") or kwargs.get("domain"))
        thing_name = self._normalize(slots.get("name") or slots.get("device") or kwargs.get("name"))

        by_area = self._collect_area_entities(area_name, domain)
        by_name = self._collect_by_name(thing_name, domain)

        # preserve order, remove dups
        merged = list(dict.fromkeys([*by_area, *by_name]))
        return {"resolved_ids": merged}
