import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.const import (
    UnitOfTemperature,
    UnitOfPower,
    UnitOfEnergy,
    PERCENTAGE,
)

from .base import Capability
from ..utils.fuzzy_utils import get_fuzz

_LOGGER = logging.getLogger(__name__)

DEVICE_CLASS_UNITS = {
    "temperature": {
        UnitOfTemperature.CELSIUS,
        UnitOfTemperature.FAHRENHEIT,
        UnitOfTemperature.KELVIN,
    },
    "power": {UnitOfPower.WATT, UnitOfPower.KILO_WATT},
    "energy": {UnitOfEnergy.WATT_HOUR, UnitOfEnergy.KILO_WATT_HOUR},
    "humidity": {PERCENTAGE},
    "battery": {PERCENTAGE},
    "illuminance": {"lx", "lm"},
    "pressure": {"hPa", "mbar", "bar", "psi"},
}

GENERIC_NAMES = {
    "licht",
    "lichter",
    "lampe",
    "lampen",
    "leuchte",
    "leuchten",
    "gerät",
    "geräte",
    "ding",
    "alles",
    "alle",
    "etwas",
}


class EntityResolverCapability(Capability):
    name = "entity_resolver"
    description = (
        "Resolve entities from NLU slots; enrich with area/floor + fuzzy matching."
    )

    _FUZZ_STRONG = 92
    _FUZZ_FALLBACK = 84
    _FUZZ_MAX_ADD = 4

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.memory = None  # Will be set by Stage1

    def set_memory(self, memory_cap):
        """Allow Stage1 to inject memory capability for alias resolution"""
        self.memory = memory_cap

    def _all_entities(self) -> Dict[str, Any]:
        ent_reg = er.async_get(self.hass)
        all_entities: Dict[str, Any] = {
            e.entity_id: e for e in ent_reg.entities.values() if not e.disabled_by
        }
        for st in self.hass.states.async_all():
            if st.entity_id not in all_entities:
                all_entities[st.entity_id] = None
        return all_entities

    async def run(
        self, user_input, *, entities: Dict[str, Any] | None = None, **_: Any
    ) -> Dict[str, Any]:
        hass: HomeAssistant = self.hass
        slots = entities or {}

        domain = self._first_str(slots, "domain", "domain_name")
        target_device_class = self._first_str(slots, "device_class")
        thing_name = self._first_str(slots, "name", "device", "entity", "label")
        raw_entity_id = self._first_str(slots, "entity_id")
        area_hint = self._first_str(slots, "area", "room")
        floor_hint = self._first_str(slots, "floor", "level")

        # === Memory-based alias resolution (before fuzzy matching) ===
        if area_hint and self.memory:
            memory_area = await self.memory.get_area_alias(area_hint)
            if memory_area:
                _LOGGER.debug(
                    "[EntityResolver] Memory hit: '%s' → '%s'", area_hint, memory_area
                )
                area_hint = memory_area  # Use memory-mapped value

        if floor_hint and self.memory:
            memory_floor = await self.memory.get_floor_alias(floor_hint)
            if memory_floor:
                _LOGGER.debug(
                    "[EntityResolver] Memory hit (floor): '%s' → '%s'",
                    floor_hint,
                    memory_floor,
                )
                floor_hint = memory_floor
        # === End memory resolution ===

        if thing_name and thing_name.lower().strip() in GENERIC_NAMES:
            _LOGGER.debug("[EntityResolver] Ignoring generic name '%s'.", thing_name)
            thing_name = None

        resolved: List[str] = []
        seen: Set[str] = set()

        if raw_entity_id and self._state_exists(raw_entity_id):
            resolved.append(raw_entity_id)
            seen.add(raw_entity_id)

        area_obj = self._find_area(area_hint) if area_hint else None
        floor_obj = self._find_floor(floor_hint) if floor_hint else None

        # Area-based Lookup
        area_entities: List[str] = []
        if area_obj:
            area_entities = self._entities_in_area(area_obj, domain)
            if not thing_name:
                for eid in area_entities:
                    if eid not in seen:
                        resolved.append(eid)
                        seen.add(eid)

        # Name-based Lookup
        if thing_name:
            all_entities = self._all_entities()
            exact = self._collect_by_name_exact(hass, thing_name, domain, all_entities)
            if area_entities:
                exact = [e for e in exact if e in set(area_entities)]

            for eid in exact:
                if eid not in seen:
                    resolved.append(eid)
                    seen.add(eid)

            fuzz = await get_fuzz()
            allowed = set(area_entities) if area_entities else None
            fuzzy_added = self._collect_by_name_fuzzy(
                hass, thing_name, domain, fuzz, all_entities, allowed=allowed
            )
            for eid in fuzzy_added:
                if eid not in seen:
                    resolved.append(eid)
                    seen.add(eid)

        # "All Domain" Fallback
        if not thing_name and not area_hint and domain:
            _LOGGER.debug(
                "[EntityResolver] No name/area specified. Fetching ALL entities for domain '%s'",
                domain,
            )
            all_domain_entities = self._collect_all_domain_entities(domain)
            for eid in all_domain_entities:
                if eid not in seen:
                    resolved.append(eid)
                    seen.add(eid)

        # Filter by Floor
        if floor_obj:
            resolved = [
                eid
                for eid in resolved
                if self._is_entity_on_floor(eid, floor_obj.floor_id)
            ]

        # Filter by Device Class
        if target_device_class:
            resolved = [
                eid
                for eid in resolved
                if self._match_device_class_or_unit(eid, target_device_class)
            ]

        # Filter Exposure
        pre_count = len(resolved)
        resolved = [
            eid
            for eid in resolved
            if async_should_expose(hass, CONVERSATION_DOMAIN, eid)
        ]

        # Phase 1: Filter by Knowledge Graph usability (dependencies)
        # Entities with unmet dependencies (e.g., Ambilight when TV off) are filtered
        filtered_by_deps = []
        try:
            from ..utils.knowledge_graph import get_knowledge_graph
            graph = get_knowledge_graph(hass)
            resolved, filtered_by_deps = graph.filter_candidates_by_usability(resolved)
            
            if filtered_by_deps:
                _LOGGER.debug(
                    "[EntityResolver] Filtered %d entities with unmet dependencies: %s",
                    len(filtered_by_deps), filtered_by_deps
                )
        except Exception as e:
            _LOGGER.debug("[EntityResolver] Knowledge graph filtering failed: %s", e)

        _LOGGER.debug(
            "[EntityResolver] Final Result: %d entities (pre-filter: %d, filtered by deps: %d)",
            len(resolved),
            pre_count,
            len(filtered_by_deps),
        )
        return {"resolved_ids": resolved, "filtered_by_deps": filtered_by_deps}

    # --- NEW HELPER ---
    def _entities_in_area_by_name(
        self, area_name: str, domain: str = None
    ) -> List[Dict[str, str]]:
        """Return list of {id, friendly_name} for all entities in an area."""
        area = self._find_area(area_name)
        if not area:
            return []

        eids = self._entities_in_area(area, domain)
        results = []
        for eid in eids:
            st = self.hass.states.get(eid)
            if st:
                name = st.attributes.get("friendly_name") or eid
                results.append({"id": eid, "friendly_name": name})
        return results

    # ----------- Existing Helpers -----------
    @staticmethod
    def _first_str(d: Dict[str, Any], *keys: str) -> Optional[str]:
        for k in keys:
            v = d.get(k)
            if isinstance(v, dict):
                v = v.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _state_exists(self, entity_id: str) -> bool:
        return self.hass.states.get(entity_id) is not None

    def _collect_all_domain_entities(self, domain: str) -> List[str]:
        return [
            state.entity_id
            for state in self.hass.states.async_all()
            if state.entity_id.startswith(f"{domain}.")
        ]

    def _find_floor(self, floor_name: str):
        if not floor_name:
            return None
        floor_reg = fr.async_get(self.hass)
        needle = self._canon(floor_name)
        for floor in floor_reg.async_list_floors():
            if self._canon(floor.name) == needle:
                return floor
        return None

    def _is_entity_on_floor(self, entity_id: str, floor_id: str) -> bool:
        ent_reg = er.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        entry = ent_reg.async_get(entity_id)
        area_id = None
        if entry:
            area_id = entry.area_id
            if not area_id and entry.device_id:
                dev = dev_reg.async_get(entry.device_id)
                if dev:
                    area_id = dev.area_id
        if not area_id:
            return False
        area = area_reg.async_get_area(area_id)
        if area and area.floor_id == floor_id:
            return True
        return False

    def _match_device_class_or_unit(self, entity_id: str, target_class: str) -> bool:
        if not target_class:
            return True
        target_class = target_class.lower().strip()
        domain = entity_id.split(".", 1)[0].lower()
        if target_class == domain:
            return True
        if target_class == "light" and domain in ("light", "switch", "input_boolean"):
            return True
        state = self.hass.states.get(entity_id)
        if not state:
            return False
        dc = state.attributes.get("device_class")
        if dc and dc.lower() == target_class:
            return True
        unit = state.attributes.get("unit_of_measurement")
        expected_units = DEVICE_CLASS_UNITS.get(target_class)
        if unit and expected_units and unit in expected_units:
            return True
        return False

    @staticmethod
    def _looks_like_entity_id(text: str) -> bool:
        s = text.strip().lower()
        return "." in s and re.match(r"^[a-z0-9_]+\.[a-z0-9_]+$", s) is not None

    @staticmethod
    def _canon(s: Optional[str]) -> str:
        if not s:
            return ""
        t = s.lower()
        t = (
            t.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        t = re.sub(r"[^\w\s]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _obj_id(eid: str) -> str:
        return eid.split(".", 1)[1] if "." in eid else eid

    def _find_area(self, area_name: Optional[str]):
        if not area_name:
            return None
        area_reg = ar.async_get(self.hass)
        needle = self._canon(area_name)
        areas = area_reg.async_list_areas()
        for a in areas:
            canon_name = self._canon(a.name or "")
            if canon_name == needle:
                return a
        return None

    def _entities_in_area(self, area, domain: Optional[str]) -> List[str]:
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)
        canon_area = self._canon(area.name)
        out: List[str] = []
        for ent in ent_reg.entities.values():
            dev = dev_reg.devices.get(ent.device_id) if ent.device_id else None
            has_any_area = bool(ent.area_id or (dev and dev.area_id))
            in_area = ent.area_id == area.id or (
                dev is not None and dev.area_id == area.id
            )
            if not in_area and not has_any_area:
                name_match = self._canon(ent.original_name or "")
                eid_match = self._canon(ent.entity_id)
                if canon_area and (canon_area in name_match or canon_area in eid_match):
                    in_area = True
            if not in_area:
                continue
            if domain and ent.domain != domain:
                continue
            out.append(ent.entity_id)
        return out

    def _collect_by_name_exact(self, hass, name, domain, all_entities) -> List[str]:
        if not name:
            return []
        needle = self._canon(name)
        out: List[str] = []
        for eid, ent in all_entities.items():
            if domain and not eid.startswith(f"{domain}."):
                continue
            st = hass.states.get(eid)
            friendly = st and st.attributes.get("friendly_name")
            if isinstance(friendly, str) and self._canon(friendly) == needle:
                out.append(eid)
                continue
            if self._canon(self._obj_id(eid)) == needle:
                out.append(eid)
        return out

    def _collect_by_name_fuzzy(
        self, hass, name, domain, fuzz_mod, all_entities, allowed=None
    ) -> List[str]:
        needle = self._canon(name)
        if not needle:
            return []
        scored: List[Tuple[str, int, str]] = []
        for eid, ent in all_entities.items():
            if not eid.startswith(f"{domain}."):
                continue
            if allowed is not None and eid not in allowed:
                continue
            st = hass.states.get(eid)
            friendly = st and st.attributes.get("friendly_name")
            cand1 = self._canon(friendly) if isinstance(friendly, str) else ""
            cand2 = self._canon(self._obj_id(eid))
            s1 = fuzz_mod.token_set_ratio(needle, cand1) if cand1 else 0
            s2 = fuzz_mod.token_set_ratio(needle, cand2) if cand2 else 0
            score = max(s1, s2)
            if score >= self._FUZZ_STRONG or score >= self._FUZZ_FALLBACK:
                label = friendly or eid
                scored.append((eid, score, label))
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[1], len(str(x[2]))))
        top = [eid for (eid, _, _) in scored[: self._FUZZ_MAX_ADD]]
        return top
