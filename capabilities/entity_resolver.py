import asyncio
import importlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import area_registry as ar, device_registry as dr, entity_registry as er
from homeassistant.helpers import entity_exposure

from .base import Capability

_LOGGER = logging.getLogger(__name__)

# ---------- non-blocking lazy import for rapidfuzz.fuzz ----------
_fuzz = None
async def _get_fuzz():
    """Lazy-load rapidfuzz.fuzz on a worker thread to avoid blocking the event loop."""
    global _fuzz
    if _fuzz is not None:
        return _fuzz
    loop = asyncio.get_running_loop()
    def _load():
        return importlib.import_module("rapidfuzz.fuzz")
    _fuzz = await loop.run_in_executor(None, _load)
    return _fuzz


class EntityResolverCapability(Capability):
    """
    Resolve entity_ids from HA NLU slots and ENRICH with:
      - area + domain lookup
      - exact friendly/object_id match
      - fuzzy match within domain (optionally restricted to area)
    """

    name = "entity_resolver"
    description = "Resolve entities from NLU slots; enrich with area + exact/fuzzy matching."

    _FUZZ_STRONG = 92
    _FUZZ_FALLBACK = 84
    _FUZZ_MAX_ADD = 4

    # ---------- NEW: unified entity registry + states ----------
    def _all_entities(self) -> Dict[str, Any]:
        """Collect all known entities (registry + state machine)."""
        ent_reg = er.async_get(self.hass)
        all_entities: Dict[str, Any] = {
            e.entity_id: e for e in ent_reg.entities.values() if not e.disabled_by
        }
        for st in self.hass.states.async_all():
            if st.entity_id not in all_entities:
                all_entities[st.entity_id] = None
        _LOGGER.debug("[EntityResolver] _all_entities() collected %d total entities", len(all_entities))
        return all_entities

    async def run(self, user_input, *, entities: Dict[str, Any] | None = None, **_: Any) -> Dict[str, Any]:
        hass: HomeAssistant = self.hass
        slots = entities or {}

        # Extract likely slots
        domain = self._first_str(slots, "domain", "domain_name")
        thing_name = self._first_str(slots, "name", "device", "entity", "label")
        raw_entity_id = self._first_str(slots, "entity_id")
        area_hint = self._first_str(slots, "area", "room")

        _LOGGER.debug(
            "[EntityResolver] Incoming slots: domain=%r, name=%r, entity_id=%r, area=%r",
            domain, thing_name, raw_entity_id, area_hint,
        )

        resolved: List[str] = []
        seen: Set[str] = set()

        # 0) Direct entity_id (verbatim)
        if raw_entity_id and self._state_exists(raw_entity_id):
            resolved.append(raw_entity_id)
            seen.add(raw_entity_id)
            _LOGGER.debug("[EntityResolver] Added direct entity_id=%s", raw_entity_id)

        # Prepare area restriction (optional)
        area_entities: List[str] = []
        area_obj = self._find_area(area_hint) if area_hint else None
        if area_obj:
            area_entities = self._entities_in_area(area_obj, domain)
            if area_entities:
                _LOGGER.debug(
                    "[EntityResolver] Area '%s' → %d entities%s",
                    area_obj.name, len(area_entities),
                    f" (domain={domain})" if domain else "",
                )

        # 1) If 'name' looks like entity_id ("domain.object"), accept it
        if thing_name and self._looks_like_entity_id(thing_name):
            eid = thing_name.strip().lower()
            if self._state_exists(eid) and eid not in seen:
                if not area_entities or eid in set(area_entities):
                    resolved.append(eid)
                    seen.add(eid)
                    _LOGGER.debug("[EntityResolver] Added entity_id parsed from name=%s", eid)

        # 2) Exact friendly-name or object_id match (merged registry + states)
        all_entities = self._all_entities()
        exact = self._collect_by_name_exact(hass, thing_name, domain, all_entities)
        if area_entities:
            exact = [e for e in exact if e in set(area_entities)]
        for eid in exact:
            if eid not in seen:
                resolved.append(eid)
                seen.add(eid)
        if exact:
            _LOGGER.debug("[EntityResolver] Exact-name/object_id matches: %s", ", ".join(exact))

        # 3) Area+domain fallback when NO name was provided
        added_area = []
        if area_entities and not thing_name:
            for eid in area_entities:
                if eid not in seen:
                    resolved.append(eid)
                    seen.add(eid)
                    added_area.append(eid)
            if added_area:
                _LOGGER.debug("[EntityResolver] Added area-based targets (no name): %s", ", ".join(added_area))

        # 4) Fuzzy enrichment within domain (optionally restricted to area)
        fuzzy_added: List[str] = []
        if domain and thing_name:
            fuzz_mod = await _get_fuzz()
            allowed = set(area_entities) if area_entities else None
            fuzzy_added = self._collect_by_name_fuzzy(hass, thing_name, domain, fuzz_mod, all_entities, allowed=allowed)
            for eid in fuzzy_added:
                if eid not in seen:
                    resolved.append(eid)
                    seen.add(eid)
            if fuzzy_added:
                _LOGGER.debug("[EntityResolver] Fuzzy enriched with: %s", ", ".join(fuzzy_added))

        # ---------- NEW: filter to exposed entities right before returning ----------
        try:
            expo = entity_exposure.get_exposed_entities(hass)
            # handle either a set-like or dict-like return without importing its internals
            if isinstance(expo, dict):
                exposed_ids: Optional[Set[str]] = set(expo.keys())
            else:
                exposed_ids = set(expo) if expo is not None else None
        except Exception as err:
            _LOGGER.warning("Entity exposure lookup failed (%s); returning unfiltered", err)
            exposed_ids = None

        if exposed_ids is not None:
            pre_count = len(resolved)
            resolved = [eid for eid in resolved if eid in exposed_ids]
            _LOGGER.debug(
                "[EntityResolver] Exposure filter kept %d/%d entities",
                len(resolved), pre_count
            )
        # ---------------------------------------------------------------------------

        _LOGGER.debug(
            "[EntityResolver] Result → %d entities (area_add=%d, exact=%d, fuzzy=%d): %s",
            len(resolved), len(added_area), len(exact), len(fuzzy_added), ", ".join(resolved),
        )
        return {"resolved_ids": resolved}

    # ----------- helpers -----------

    @staticmethod
    def _first_str(d: Dict[str, Any], *keys: str) -> Optional[str]:
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _state_exists(self, entity_id: str) -> bool:
        return self.hass.states.get(entity_id) is not None

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

    # ---- area helpers ----
    def _find_area(self, area_name: Optional[str]):
        if not area_name:
            return None
        area_reg = ar.async_get(self.hass)
        needle = self._canon(area_name)
        for a in area_reg.async_list_areas():
            if self._canon(a.name or "") == needle:
                return a
        _LOGGER.debug("[EntityResolver] Area '%s' not found (canon=%r)", area_name, needle)
        return None

    def _entities_in_area(self, area, domain: Optional[str]) -> List[str]:
        """Return all entities in or textually related to an area, optionally filtered by domain.

        Fallback: if an entity has no assigned area, but its friendly name or entity_id
        contains the area name (e.g. 'hauswirtschaftsraum'), include it.
        """
        dev_reg = dr.async_get(self.hass)
        ent_reg = er.async_get(self.hass)
        device_ids: Set[str] = {d.id for d in dev_reg.devices.values() if d.area_id == area.id}
        canon_area = self._canon(area.name)

        out: List[str] = []
        for ent in ent_reg.entities.values():
            # explicit area assignment or via device area
            in_area = (ent.area_id == area.id) or (ent.device_id in device_ids if ent.device_id else False)

            # --- NEW: fuzzy textual fallback if area not assigned ---
            if not in_area:
                name_match = self._canon(ent.original_name or "")
                eid_match = self._canon(ent.entity_id)
                if canon_area and (canon_area in name_match or canon_area in eid_match):
                    in_area = True

            if not in_area:
                continue
            if domain and ent.domain != domain:
                continue
            out.append(ent.entity_id)

        _LOGGER.debug(
            "[EntityResolver] _entities_in_area('%s') → %d entity(ies) (domain=%s)",
            area.name, len(out), domain or "any"
        )
        return out

    # ---- exact/fuzzy name matching ----
    def _collect_by_name_exact(
        self,
        hass: HomeAssistant,
        name: Optional[str],
        domain: Optional[str],
        all_entities: Dict[str, Any],
    ) -> List[str]:
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
        self,
        hass: HomeAssistant,
        name: str,
        domain: str,
        fuzz_mod,
        all_entities: Dict[str, Any],
        *,
        allowed: Optional[Set[str]] = None,
    ) -> List[str]:
        """Fuzzy-match within a domain; optionally restrict to allowed entity_ids."""
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
        _LOGGER.debug(
            "[EntityResolver] Fuzzy match for %r (domain=%s%s) → %d hit(s): %s",
            name,
            domain,
            f", restricted to area ({len(allowed)} items)" if allowed is not None else "",
            len(top),
            ", ".join(top),
        )
        return top
