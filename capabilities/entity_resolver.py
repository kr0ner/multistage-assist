"""Entity resolver capability.

Resolves entities from NLU slots using:
- Area/floor-based lookup
- Name matching (exact and fuzzy)
- Device class filtering
- Knowledge graph dependency filtering
- Capability filtering (e.g., dimmability)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN

from .base import Capability
from .area_resolver import AreaResolverCapability
from ..utils.fuzzy_utils import get_fuzz
from ..utils.german_utils import canonicalize
from ..constants.entity_keywords import GENERIC_NAMES, ALL_KEYWORDS
from ..constants.sensor_units import DEVICE_CLASS_UNITS

_LOGGER = logging.getLogger(__name__)


class EntityResolverCapability(Capability):
    """Resolve entities from NLU slots with area/floor and fuzzy matching."""
    
    name = "entity_resolver"
    description = "Resolve entities from NLU slots; enrich with area/floor + fuzzy matching."

    _FUZZ_STRONG = 92
    _FUZZ_FALLBACK = 84
    _FUZZ_MAX_ADD = 4

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.memory = None  # Injected by caller
        self._area_resolver = AreaResolverCapability(hass, config)

    def set_memory(self, memory_cap):
        """Inject memory capability for alias resolution."""
        self.memory = memory_cap

    def _all_entities(self) -> Dict[str, Any]:
        """Get all non-disabled entities."""
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
        """Resolve entities from NLU slot data.
        
        Args:
            user_input: ConversationInput
            entities: Dict of NLU slots
            
        Returns:
            Dict with "resolved_ids" and "filtered_by_deps"
        """
        hass: HomeAssistant = self.hass
        slots = entities or {}

        domain = self._first_str(slots, "domain", "domain_name")
        target_device_class = self._first_str(slots, "device_class")
        thing_name = self._first_str(slots, "name", "device", "entity", "label")
        raw_entity_id = self._first_str(slots, "entity_id")
        area_hint = self._first_str(slots, "area", "room")
        floor_hint = self._first_str(slots, "floor", "level")

        # === Memory-based alias resolution ===
        if area_hint and self.memory:
            memory_area = await self.memory.get_area_alias(area_hint)
            if memory_area:
                _LOGGER.debug("[EntityResolver] Memory hit: '%s' → '%s'", area_hint, memory_area)
                area_hint = memory_area

        if floor_hint and self.memory:
            memory_floor = await self.memory.get_floor_alias(floor_hint)
            if memory_floor:
                _LOGGER.debug("[EntityResolver] Memory hit (floor): '%s' → '%s'", floor_hint, memory_floor)
                floor_hint = memory_floor

        # Ignore generic names
        if thing_name and thing_name.lower().strip() in GENERIC_NAMES:
            _LOGGER.debug("[EntityResolver] Ignoring generic name '%s'.", thing_name)
            thing_name = None

        resolved: List[str] = []
        seen: Set[str] = set()

        # Direct entity ID
        if raw_entity_id and self._state_exists(raw_entity_id):
            resolved.append(raw_entity_id)
            seen.add(raw_entity_id)

        # Use area_resolver for area/floor lookup
        area_obj = self._area_resolver.find_area(area_hint) if area_hint else None
        floor_obj = self._area_resolver.find_floor(floor_hint) if floor_hint else None

        # Area-based lookup
        area_entities: List[str] = []
        if area_obj:
            area_entities = self._entities_in_area(area_obj, domain)
            if not thing_name:
                for eid in area_entities:
                    if eid not in seen:
                        resolved.append(eid)
                        seen.add(eid)

        # Name-based lookup
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

        # "All Domain" fallback
        if not thing_name and not area_hint and domain:
            # CRITICAL SAFETY CHECK: Only allow global "turn everything on/off"
            # if the user explicitly said "all/sämtliche".
            # Otherwise "Schalte Spots an" -> "Spot" is generic -> triggers this -> turns on whole house.
            has_all_keyword = any(k in user_input.text.lower() for k in ALL_KEYWORDS)
            
            if has_all_keyword:
                _LOGGER.debug("[EntityResolver] No name/area. Fetching ALL entities for domain '%s'", domain)
                all_domain_entities = self._collect_all_domain_entities(domain)
                for eid in all_domain_entities:
                    if eid not in seen:
                        resolved.append(eid)
                        seen.add(eid)
            else:
                _LOGGER.debug("[EntityResolver] Global fallback skipped (no 'all' keyword in '%s')", user_input.text)

        # Filter by floor
        if floor_obj:
            before_floor = resolved.copy()
            resolved = [
                eid for eid in resolved
                if self._is_entity_on_floor(eid, floor_obj.floor_id)
            ]
            filtered_by_floor = [eid for eid in before_floor if eid not in resolved]
            if filtered_by_floor:
                _LOGGER.debug("[EntityResolver] Filtered %d by floor: %s", len(filtered_by_floor), filtered_by_floor)

        # Filter by device class
        if target_device_class:
            before_class = resolved.copy()
            resolved = [
                eid for eid in resolved
                if self._match_device_class_or_unit(eid, target_device_class)
            ]
            filtered_by_class = [eid for eid in before_class if eid not in resolved]
            if filtered_by_class:
                _LOGGER.debug("[EntityResolver] Filtered %d by device_class: %s", len(filtered_by_class), filtered_by_class)

        # Filter by exposure (entities must be exposed to conversation agent)
        pre_count = len(resolved)
        before_expose = resolved.copy()
        resolved = [
            eid for eid in resolved
            if async_should_expose(hass, CONVERSATION_DOMAIN, eid)
        ]
        filtered_by_expose = [eid for eid in before_expose if eid not in resolved]
        if filtered_by_expose:
            _LOGGER.debug(
                "[EntityResolver] Filtered %d NOT EXPOSED to conversation: %s",
                len(filtered_by_expose), filtered_by_expose
            )

        # Filter by knowledge graph dependencies
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

        # Filter by capability (e.g., dimmability for HassLightSet)
        intent = self._first_str(slots, "intent")
        if intent == "HassLightSet" and domain == "light":
            before_cap = len(resolved)
            resolved = [eid for eid in resolved if self._is_light_dimmable(eid)]
            if before_cap != len(resolved):
                _LOGGER.debug(
                    "[EntityResolver] Filtered %d non-dimmable lights for HassLightSet",
                    before_cap - len(resolved)
                )

        _LOGGER.debug(
            "[EntityResolver] Final: %d entities (pre-filter: %d, filtered by deps: %d, not exposed: %d)",
            len(resolved), pre_count, len(filtered_by_deps), len(filtered_by_expose),
        )
        return {
            "resolved_ids": resolved,
            "filtered_by_deps": filtered_by_deps,
            "filtered_not_exposed": filtered_by_expose,
        }


    # --- Helper Methods ---
    
    def _is_light_dimmable(self, entity_id: str) -> bool:
        """Check if a light entity supports dimming."""
        state = self.hass.states.get(entity_id)
        if not state:
            return False
        modes = state.attributes.get("supported_color_modes", [])
        return not modes or modes != ["onoff"]

    def _entities_in_area_by_name(self, area_name: str, domain: str = None) -> List[Dict[str, str]]:
        """Return list of {id, friendly_name} for all entities in an area."""
        area = self._area_resolver.find_area(area_name)
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

    @staticmethod
    def _first_str(d: Dict[str, Any], *keys: str) -> Optional[str]:
        """Extract first string value from dict."""
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

    def _is_entity_on_floor(self, entity_id: str, floor_id: str) -> bool:
        """Check if entity is on the specified floor."""
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
        return area and area.floor_id == floor_id

    def _match_device_class_or_unit(self, entity_id: str, target_class: str) -> bool:
        """Match entity by device class or unit of measurement."""
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
        import re
        s = text.strip().lower()
        return "." in s and re.match(r"^[a-z0-9_]+\.[a-z0-9_]+$", s) is not None

    @staticmethod
    def _obj_id(eid: str) -> str:
        return eid.split(".", 1)[1] if "." in eid else eid

    def _entities_in_area(self, area, domain: Optional[str]) -> List[str]:
        """Get all entities in an area."""
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)
        canon_area = canonicalize(area.name)
        out: List[str] = []
        
        for ent in ent_reg.entities.values():
            dev = dev_reg.devices.get(ent.device_id) if ent.device_id else None
            has_any_area = bool(ent.area_id or (dev and dev.area_id))
            in_area = ent.area_id == area.id or (dev is not None and dev.area_id == area.id)
            
            if not in_area and not has_any_area:
                name_match = canonicalize(ent.original_name or "")
                eid_match = canonicalize(ent.entity_id)
                if canon_area and (canon_area in name_match or canon_area in eid_match):
                    in_area = True
                    
            if not in_area:
                continue
            if domain and ent.domain != domain:
                continue
            out.append(ent.entity_id)
        return out

    def _collect_by_name_exact(self, hass, name, domain, all_entities) -> List[str]:
        """Collect entities by exact name match."""
        if not name:
            return []
        needle = canonicalize(name)
        out: List[str] = []
        ent_reg = er.async_get(hass)
        
        for eid, ent in all_entities.items():
            if domain and not eid.startswith(f"{domain}."):
                continue
            st = hass.states.get(eid)
            friendly = st and st.attributes.get("friendly_name")
            
            if isinstance(friendly, str) and canonicalize(friendly) == needle:
                out.append(eid)
                continue
            
            if canonicalize(self._obj_id(eid)) == needle:
                out.append(eid)
                continue
            
            entry = ent_reg.async_get(eid)
            if entry:
                aliases = getattr(entry, "aliases", None) or set()
                for alias in aliases:
                    if canonicalize(alias) == needle:
                        _LOGGER.debug("[EntityResolver] Entity alias match: '%s' → '%s'", name, eid)
                        out.append(eid)
                        break
        
        return out

    def _collect_by_name_fuzzy(
        self, hass, name, domain, fuzz_mod, all_entities, allowed=None
    ) -> List[str]:
        """Collect entities by fuzzy name match."""
        needle = canonicalize(name)
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
            cand1 = canonicalize(friendly) if isinstance(friendly, str) else ""
            cand2 = canonicalize(self._obj_id(eid))
            s1 = fuzz_mod.token_set_ratio(needle, cand1) if cand1 else 0
            s2 = fuzz_mod.token_set_ratio(needle, cand2) if cand2 else 0
            score = max(s1, s2)
            if score >= self._FUZZ_STRONG or score >= self._FUZZ_FALLBACK:
                label = friendly or eid
                scored.append((eid, score, label))
                
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[1], len(str(x[2]))))
        return [eid for (eid, _, _) in scored[:self._FUZZ_MAX_ADD]]
