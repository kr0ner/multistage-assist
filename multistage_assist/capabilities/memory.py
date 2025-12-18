import logging
from typing import Any, Dict, Optional
from homeassistant.helpers.storage import Store
from .base import Capability

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = "multistage_assist_memory"
STORAGE_VERSION = 1

class MemoryCapability(Capability):
    """
    Saves and loads aliases for Areas, Entities, and Floors.
    """
    name = "memory"
    
    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data = None # Lazy load

    async def _ensure_loaded(self):
        if self._data is None:
            self._data = await self._store.async_load() or {}
            
            # Ensure structure
            for key in ["areas", "entities", "floors"]:
                if key not in self._data:
                    self._data[key] = {}
            
            _LOGGER.debug("[Memory] Loaded data: %s", self._data)

    # --- AREAS ---
    async def get_area_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["areas"].get(text.lower().strip())

    async def learn_area_alias(self, text: str, area_name: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["areas"].get(key) != area_name:
            self._data["areas"][key] = area_name
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Area Alias: '%s' -> '%s'", key, area_name)

    # --- ENTITIES ---
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

    # --- FLOORS (NEW) ---
    async def get_floor_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["floors"].get(text.lower().strip())

    async def learn_floor_alias(self, text: str, floor_name: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["floors"].get(key) != floor_name:
            self._data["floors"][key] = floor_name
            await self._store.async_save(self._data)
            _LOGGER.info("[Memory] Learned Floor Alias: '%s' -> '%s'", key, floor_name)