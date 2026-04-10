"""Capability for managing the Home Assistant Knowledge Graph.

The Knowledge Graph centralizes:
1. Physical Device Relationships (powered_by, coupled_with)
2. Personal Data & Relationships (Memory)
3. Contextual Logic (Lux sensors, associated covers)
4. Persistent Aliases (Areas, Entities, Floors)
"""

import logging
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from homeassistant.helpers.storage import Store
from .base import Capability

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = "multistage_assist_knowledge"
LEGACY_MEMORY_KEY = "multistage_assist_memory"
STORAGE_VERSION = 1


class RelationType(Enum):
    """Types of entity relationships."""
    
    # Physical power dependency: A needs B to be on to function
    # Example: A speaker plugged into a smart switch
    POWERED_BY = "powered_by"
    
    # Logical requirement: A is only useful if B is also active
    COUPLED_WITH = "coupled_with"
    
    # Light level sensor for brightness-aware light control
    LUX_SENSOR = "lux_sensor"
    
    # Cover associated with light for hybrid natural light control
    ASSOCIATED_COVER = "associated_cover"
    
    # Parent in energy monitoring hierarchy: A is a sub-meter of B
    # This is NOT a physical power dependency.
    ENERGY_PARENT = "energy_parent"


class ActivationMode(Enum):
    """How a dependent device should be activated."""
    AUTO = "auto"           # Automatically enable dependency
    WARN = "warn"           # Warn user and suggest enabling
    SYNC = "sync"           # Keep in sync (turn on/off together)
    MANUAL = "manual"       # Just inform, don't suggest


@dataclass
class Dependency:
    """Represents a dependency between entities."""
    source_entity: str
    target_entity: str
    relation_type: RelationType
    activation_mode: ActivationMode = ActivationMode.AUTO
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyResolution:
    """Result of resolving dependencies for an action."""
    can_proceed: bool
    prerequisites: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None


class KnowledgeGraphCapability(Capability):
    """
    Unified Knowledge Graph and Memory capability.
    
    Acts as the primary store for device relationships and personal data.
    """

    name = "knowledge_graph"
    description = "A unified storage for Home Assistant context, including: 1. Physical and logical device dependencies (power/energy) 2. Personal memory (facts about users) 3. Area/Floor/Entity aliases learned over time. Acts as the persistent memory layer for context-aware reasoning."

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data = None  # Lazy load
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self):
        """Lazy load data and perform migrations if needed."""
        if self._data is not None:
            return
            
        async with self._lock:
            if self._data is not None:
                return
            
            # 1. Try to load new storage
            self._data = await self._store.async_load()
            
            # 2. Migration from Legacy Memory if new storage doesn't exist
            if not self._data:
                legacy_store = Store(self.hass, 2, LEGACY_MEMORY_KEY)
                legacy_data = await legacy_store.async_load()
                if legacy_data:
                    _LOGGER.info("[KnowledgeGraph] Migrating from Legacy Memory storage")
                    self._data = legacy_data
                    await self._store.async_save(self._data)
                else:
                    self._data = {}
            
            # 3. Ensure structure
            for key in ["areas", "entities", "floors", "personal", "relationships"]:
                if key not in self._data:
                    self._data[key] = {}
            
            _LOGGER.debug("[KnowledgeGraph] Loaded data: %s", self._data)

    async def _save(self):
        """Save data to storage."""
        async with self._lock:
            await self._store.async_save(self._data)

    # --- Alias Management (Legacy Memory functionality) ---

    async def get_area_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["areas"].get(text.lower().strip())

    async def learn_area_alias(self, text: str, area_name: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["areas"].get(key) != area_name:
            self._data["areas"][key] = area_name
            await self._save()

    async def get_entity_alias(self, text: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["entities"].get(text.lower().strip())

    async def learn_entity_alias(self, text: str, entity_id: str):
        await self._ensure_loaded()
        key = text.lower().strip()
        if self._data["entities"].get(key) != entity_id:
            self._data["entities"][key] = entity_id
            await self._save()

    # --- Personal Data ---

    async def get_personal_data(self, key: str) -> Optional[str]:
        await self._ensure_loaded()
        return self._data["personal"].get(key.lower().strip())

    async def get_all_personal_data(self) -> Dict[str, str]:
        await self._ensure_loaded()
        return self._data["personal"]

    async def learn_personal_data(self, key: str, value: str):
        await self._ensure_loaded()
        k = key.lower().strip()
        if self._data["personal"].get(k) != value:
            self._data["personal"][k] = value
            await self._save()

    # --- Relationship Resolution (Legacy KnowledgeGraph utility functionality) ---

    async def learn_dependency(self, source: str, target: str, relation: RelationType, mode: ActivationMode = ActivationMode.AUTO):
        """Learn a new dependency between entities."""
        await self._ensure_loaded()
        key = f"{source} -> {target}"
        self._data["relationships"][key] = {
            "source": source,
            "target": target,
            "relation": relation.value,
            "mode": mode.value
        }
        await self._save()
        _LOGGER.info("[KnowledgeGraph] Learned dependency: %s", key)

    async def remove_dependency(self, source: str, target: str):
        """Remove a dependency between entities."""
        await self._ensure_loaded()
        key = f"{source} -> {target}"
        if key in self._data["relationships"]:
            del self._data["relationships"][key]
            await self._save()
            _LOGGER.info("[KnowledgeGraph] Removed dependency: %s", key)

    async def get_dependencies(self, entity_id: str) -> List[Dependency]:
        """Get all dependencies for an entity from state attributes and persistent storage."""
        await self._ensure_loaded()
        all_deps = []
        
        # 1. State-based dependencies (from customize.yaml)
        state = self.hass.states.get(entity_id)
        if state:
            attrs = state.attributes
            if "powered_by" in attrs:
                all_deps.append(Dependency(entity_id, attrs["powered_by"], RelationType.POWERED_BY, 
                                          ActivationMode(attrs.get("activation_mode", "auto"))))
            if "coupled_with" in attrs:
                all_deps.append(Dependency(entity_id, attrs["coupled_with"], RelationType.COUPLED_WITH, 
                                          ActivationMode(attrs.get("activation_mode", "sync"))))
        
        # 2. Persistent dependencies (from storage)
        for rel in self._data["relationships"].values():
            if rel["source"] == entity_id:
                all_deps.append(Dependency(
                    rel["source"], 
                    rel["target"], 
                    RelationType(rel["relation"]), 
                    ActivationMode(rel["mode"])
                ))
                    
        return all_deps

    async def filter_candidates_by_usability(
        self, entity_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter a list of entities to only those that are currently usable.
        
        An entity is unusable if it has a powered_by dependency that is OFF
        and the activation mode is NOT 'auto'.
        """
        usable = []
        filtered = []
        
        for eid in entity_ids:
            is_usable, reason = await self.is_entity_usable(eid)
            if is_usable:
                usable.append(eid)
            else:
                filtered.append(eid)
                _LOGGER.debug("[KnowledgeGraph] Filtered %s: %s", eid, reason)
                
        return usable, filtered

    async def is_entity_usable(self, entity_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a single entity is usable given current dependencies."""
        deps = await self.get_dependencies(entity_id)
        for dep in deps:
            target_state = self.hass.states.get(dep.target_entity)
            if not target_state:
                continue
                
            if dep.relation_type == RelationType.POWERED_BY:
                if target_state.state in ("off", "unavailable"):
                    if dep.activation_mode != ActivationMode.AUTO:
                        return False, f"Powered by {dep.target_entity} which is {target_state.state}"
                        
            elif dep.relation_type == RelationType.COUPLED_WITH:
                # Coupled devices don't necessarily make it unusable, 
                # they just usually get synced during execution.
                pass
                
        return True, None

    async def resolve_for_action(self, entity_id: str, action: str) -> DependencyResolution:
        """Resolve prerequisites for an action."""
        # Mapping common intents to actions if needed, but executor usually passes action
        res = DependencyResolution(can_proceed=True)
        deps = await self.get_dependencies(entity_id)
        
        for dep in deps:
            target_state = self.hass.states.get(dep.target_entity)
            if not target_state: continue
            
            if dep.relation_type == RelationType.POWERED_BY:
                if action in ("turn_on", "activate", "play"):
                    if target_state.state in ("off", "unavailable"):
                        if dep.activation_mode == ActivationMode.AUTO:
                            res.prerequisites.append({"entity_id": dep.target_entity, "action": "turn_on"})
                        else:
                            res.can_proceed = False
                            res.warnings.append(f"{dep.target_entity} ist aus.")
                            
            elif dep.relation_type == RelationType.COUPLED_WITH:
                if action in ("turn_on", "activate", "play"):
                    if target_state.state in ("off", "unavailable", "idle"):
                        if dep.activation_mode == ActivationMode.SYNC:
                            res.prerequisites.append({"entity_id": dep.target_entity, "action": "turn_on"})
        
        return res

    async def run(self, user_input, **kwargs) -> Dict[str, Any]:
        """Standard capability interface."""
        return {}
