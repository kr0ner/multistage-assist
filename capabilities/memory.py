import logging
from typing import Optional, Dict
from homeassistant.helpers.storage import Store
from .base import Capability

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = "multistage_assist_memory"
STORAGE_VERSION = 1

class MemoryCapability(Capability):
    """
    Stores learned aliases for Areas AND Entities.
    Structure: {"areas": {"bad": "badezimmer"}, "entities": {"spiegellicht": "light.bad_spiegel"}}
    """
    name = "memory"
    
    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data = None

    async def _ensure_loaded(self):
        if self._data is None:
            self._data = await self._store.async_load() or {"areas": {}, "entities": {}}
            # Ensure structure structure
            if "entities" not in self._data: self._data["entities"] = {}

    async def get_area_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["areas"].get(text.lower().strip())

    async def learn_area_alias(self, text: str, area_name: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["areas"].get(key) != area_name:
            self._data["areas"][key] = area_name
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Area: '%s' -> '%s'", key, area_name)

    async def get_entity_alias(self, text: str) -> Optional[str]:
        """Get mapped entity_id for a specific name."""
        await self._ensure_loaded()
        return self._data["entities"].get(text.lower().strip())

    async def learn_entity_alias(self, text: str, entity_id: str):
        """Learn that a specific name refers to a specific entity_id."""
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["entities"].get(key) != entity_id:
            self._data["entities"][key] = entity_id
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Entity: '%s' -> '%s'", key, entity_id)