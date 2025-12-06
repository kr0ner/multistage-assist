import logging
from typing import Any, Dict, Optional
from homeassistant.helpers.storage import Store
from .base import Capability

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = "multistage_assist_memory"
STORAGE_VERSION = 1

class MemoryCapability(Capability):
    """
    Saves and loads aliases (e.g. 'Bad' -> 'Badezimmer').
    """
    name = "memory"
    
    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data = None # Lazy load

    async def _ensure_loaded(self):
        if self._data is None:
            self._data = await self._store.async_load() or {"areas": {}, "entities": {}}
            # Ensure structure
            if "areas" not in self._data: self._data["areas"] = {}
            if "entities" not in self._data: self._data["entities"] = {}
            _LOGGER.debug("[Memory] Loaded data: %s", self._data)

    async def get_area_alias(self, text: str) -> Optional[str]:
        """Checks if an alias is available for the given input"""
        await self._ensure_loaded()
        val = self._data["areas"].get(text.lower().strip())
        if val:
            _LOGGER.debug("[Memory] Found alias for '%s' -> '%s'", text, val)
        return val

    async def learn_area_alias(self, text: str, area_name: str):
        """Saves new alias."""
        await self._ensure_loaded()
        key = text.lower().strip()
        
        if self._data["areas"].get(key) != area_name:
            self._data["areas"][key] = area_name
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Area Alias: '%s' -> '%s'", key, area_name)
        else:
            _LOGGER.debug("[Memory] Alias '%s' -> '%s' already known.", key, area_name)

    async def get_entity_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["entities"].get(text.lower().strip())

    async def learn_entity_alias(self, text: str, entity_id: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["entities"].get(key) != entity_id:
            self._data["entities"][key] = entity_id
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Entity: '%s' -> '%s'", key, entity_id)