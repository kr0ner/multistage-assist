import dataclasses
from homeassistant.helpers import area_registry, entity_registry


@dataclasses.dataclass
class ResolvedEntities:
    by_area: list[str]
    by_name: list[str]

    @property
    def merged(self) -> list[str]:
        return list({*self.by_area, *self.by_name})


class EntityResolver:
    """Utility to resolve entities from NLU slots."""

    def __init__(self, hass):
        self.hass = hass

    async def resolve(self, entities: dict[str, str]) -> ResolvedEntities:
        ent_reg = entity_registry.async_get(self.hass)
        area_reg = area_registry.async_get(self.hass)

        domain = entities.get("domain")
        area_name = entities.get("area")
        name = entities.get("name")

        by_area: list[str] = []
        by_name: list[str] = []

        for ent in ent_reg.entities.values():
            if domain and ent.domain != domain:
                continue

            if area_name:
                if ent.area_id:
                    area = area_reg.async_get_area(ent.area_id)
                    if area and area.name.lower() == area_name.lower():
                        by_area.append(ent.entity_id)

            if name and ent.original_name and name.lower() in ent.original_name.lower():
                by_name.append(ent.entity_id)

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
