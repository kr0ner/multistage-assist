import logging
from typing import Dict, Type, List, Any
from .capabilities.base import Capability

_LOGGER = logging.getLogger(__name__)


class BaseStage:
    """Base stage orchestrator managing a collection of capabilities."""

    name = "base"
    capabilities: List[Type[Capability]] = []

    def __init__(self, hass, config):
        self.hass = hass
        self.config = config
        self.capabilities_map: Dict[str, Capability] = {
            cap.name: cap(hass, config) for cap in self.capabilities
        }

    def has(self, name: str) -> bool:
        return name in self.capabilities_map

    def get(self, name: str) -> Capability:
        if name not in self.capabilities_map:
            raise KeyError(f"Capability '{name}' not found in stage {self.name}")
        return self.capabilities_map[name]

    async def use(self, name: str, user_input, **kwargs) -> Any:
        cap = self.get(name)
        _LOGGER.debug("[%s] Using capability '%s' with kwargs=%s", self.name, name, list(kwargs.keys()))
        result = await cap.run(user_input, **kwargs)
        _LOGGER.debug("[%s] Capability '%s' returned: %s", self.name, name, result)
        return result
