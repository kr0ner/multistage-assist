# entity_resolver.py
import dataclasses
import asyncio
import importlib
from homeassistant.helpers import area_registry, entity_registry, device_registry

_fuzz = None


async def get_fuzz():
    """Lazy-load rapidfuzz.fuzz in executor to avoid blocking event loop."""
    global _fuzz
    if _fuzz is None:
        loop = asyncio.get_running_loop()

        def _load():
            return importlib.import_module("rapidfuzz.fuzz")

        _fuzz = await loop.run_in_executor(None, _load)
    return _fuzz


@dataclasses.dataclass
class ResolvedEntities:
    by_area: list[str]
    by_name: list[str]

    @property
    def merged(self) -> list[str]:
        return list({*self.by_area, *self.by_name})


class EntityResolver:
    """Utility to resolve entities from NLU slots with fuzzy matching."""

    def __init__(self, hass, threshold: int = 90):
        self.hass = hass
        self.threshold = threshold  # min similarity percentage

    async def resolve(self, entities: dict[str, str]) -> ResolvedEntities:
        ent_reg = entity_registry.async_get(self.hass)
        area_reg = area_registry.async_get(self.hass)
        dev_reg = device_registry.async_get(self.hass)

        domain = entities.get("domain")
        area_name = entities.get("area")
        name = entities.get("name")
        measurement = entities.get("measurement")

        by_area: list[str] = []
        by_name: list[str] = []

        norm_area = area_name.strip().lower() if area_name else None
        norm_name = name.strip().lower() if name else None

        fuzz = await get_fuzz()

        for ent in ent_reg.entities.values():
            # Domain filter
            if domain and ent.domain != domain:
                continue

            state = self.hass.states.get(ent.entity_id)

            # Measurement filter for sensors
            if domain == "sensor" and measurement:
                if not state or state.attributes.get("device_class") != measurement:
                    continue

            # Determine area name: prefer entity.area_id, else device.area_id
            area_name_candidate = None
            area_obj = None
            if ent.area_id:
                area_obj = area_reg.async_get_area(ent.area_id)
            elif ent.device_id:
                dev = dev_reg.async_get(ent.device_id)
                if dev and dev.area_id:
                    area_obj = area_reg.async_get_area(dev.area_id)
            if area_obj:
                area_name_candidate = area_obj.name

            # Area fuzzy match
            if norm_area and area_name_candidate:
                score = fuzz.ratio(norm_area, area_name_candidate.lower())
                if score >= self.threshold:
                    by_area.append(ent.entity_id)

            # Name fuzzy match (original_name OR current state name)
            if norm_name:
                candidates = []
                if ent.original_name:
                    candidates.append(ent.original_name.lower())
                if state and state.name:
                    candidates.append(state.name.lower())
                for cand in candidates:
                    if fuzz.ratio(norm_name, cand) >= self.threshold:
                        by_name.append(ent.entity_id)
                        break

            # Fallbacks (unchanged)
            if domain and not norm_area and not norm_name:
                by_area.append(ent.entity_id)
            if norm_area and not domain and not norm_name and area_name_candidate:
                if fuzz.ratio(norm_area, area_name_candidate.lower()) >= self.threshold:
                    by_area.append(ent.entity_id)

        return ResolvedEntities(by_area=by_area, by_name=by_name)

    async def make_entity_map(self, entity_ids: list[str]) -> dict[str, str]:
        ent_reg = entity_registry.async_get(self.hass)
        states = self.hass.states

        entity_map = {}
        for eid in entity_ids:
            ent = ent_reg.async_get(eid)
            state = states.get(eid)
            if state and state.name:
                entity_map[eid] = state.name
            elif ent and ent.original_name:
                entity_map[eid] = ent.original_name
            else:
                entity_map[eid] = eid
        return entity_map
