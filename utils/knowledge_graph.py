"""Knowledge Graph for Home Assistant entity relationships.

Reads custom attributes from entities to model functional dependencies,
device coupling, and contextual logic that Home Assistant doesn't natively support.

Supported Relationship Types:
    - powered_by: Power dependency (device needs another device on to work)
    - coupled_with: Device coupling (device only useful when another is active)
    - lux_sensor: Light sensor for contextual brightness decisions
    - associated_cover: Cover that can provide light instead of turning on lights
    - energy_parent: Parent in energy hierarchy (for consumption tracking)

Configuration in customize.yaml:
    homeassistant:
      customize:
        media_player.kitchen_radio:
          powered_by: switch.kitchen_main_power
        
        light.living_room_ambilight:
          coupled_with: media_player.living_room_tv
          activation_mode: sync
        
        light.living_room_main:
          lux_sensor: sensor.living_room_lux
          associated_cover: cover.living_room_blinds
          min_lux_threshold: 200
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of entity relationships."""
    POWERED_BY = "powered_by"           # Power dependency
    COUPLED_WITH = "coupled_with"       # Must be on to be useful
    LUX_SENSOR = "lux_sensor"           # Light level sensor
    ASSOCIATED_COVER = "associated_cover"  # Cover for natural light
    ENERGY_PARENT = "energy_parent"     # Parent in energy hierarchy


class ActivationMode(Enum):
    """How a dependent device should be activated."""
    AUTO = "auto"           # Automatically enable dependency
    WARN = "warn"           # Warn user and suggest enabling
    SYNC = "sync"           # Keep in sync (turn on/off together)
    MANUAL = "manual"       # Just inform, don't suggest


@dataclass
class Dependency:
    """Represents a dependency between entities."""
    source_entity: str              # Entity that has the dependency
    target_entity: str              # Entity it depends on
    relation_type: RelationType     # Type of relationship
    activation_mode: ActivationMode = ActivationMode.AUTO
    threshold: Optional[float] = None  # For lux thresholds, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyResolution:
    """Result of resolving dependencies for an action."""
    can_proceed: bool               # Can the action proceed directly?
    prerequisites: List[Dict[str, Any]] = field(default_factory=list)  # Actions to take first
    warnings: List[str] = field(default_factory=list)  # Warnings to show user
    suggestions: List[str] = field(default_factory=list)  # Alternative suggestions
    blocked_reason: Optional[str] = None  # Why action is blocked


class KnowledgeGraph:
    """
    Knowledge Graph for entity relationships in Home Assistant.
    
    Reads custom attributes from entities to build a dependency graph
    and provides resolution logic for the conversation system.
    """
    
    # Custom attribute names
    ATTR_POWERED_BY = "powered_by"
    ATTR_COUPLED_WITH = "coupled_with"
    ATTR_ACTIVATION_MODE = "activation_mode"
    ATTR_LUX_SENSOR = "lux_sensor"
    ATTR_ASSOCIATED_COVER = "associated_cover"
    ATTR_MIN_LUX = "min_lux_threshold"
    ATTR_ENERGY_PARENT = "energy_parent"
    
    # Default thresholds
    DEFAULT_LUX_THRESHOLD = 200  # Below this, room needs light
    
    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        self._dependencies: Dict[str, List[Dependency]] = {}
        self._reverse_index: Dict[str, List[str]] = {}  # target -> [sources]
        self._cache_valid = False
    
    def refresh_graph(self) -> None:
        """Rebuild the dependency graph from entity attributes."""
        self._dependencies.clear()
        self._reverse_index.clear()
        
        for state in self.hass.states.async_all():
            entity_id = state.entity_id
            attrs = state.attributes
            
            # Check for power dependency
            if self.ATTR_POWERED_BY in attrs:
                self._add_dependency(
                    entity_id,
                    attrs[self.ATTR_POWERED_BY],
                    RelationType.POWERED_BY,
                    attrs.get(self.ATTR_ACTIVATION_MODE, "auto"),
                )
            
            # Check for device coupling
            if self.ATTR_COUPLED_WITH in attrs:
                self._add_dependency(
                    entity_id,
                    attrs[self.ATTR_COUPLED_WITH],
                    RelationType.COUPLED_WITH,
                    attrs.get(self.ATTR_ACTIVATION_MODE, "warn"),
                )
            
            # Check for lux sensor
            if self.ATTR_LUX_SENSOR in attrs:
                self._add_dependency(
                    entity_id,
                    attrs[self.ATTR_LUX_SENSOR],
                    RelationType.LUX_SENSOR,
                    threshold=attrs.get(self.ATTR_MIN_LUX, self.DEFAULT_LUX_THRESHOLD),
                )
            
            # Check for associated cover
            if self.ATTR_ASSOCIATED_COVER in attrs:
                self._add_dependency(
                    entity_id,
                    attrs[self.ATTR_ASSOCIATED_COVER],
                    RelationType.ASSOCIATED_COVER,
                )
            
            # Check for energy parent
            if self.ATTR_ENERGY_PARENT in attrs:
                self._add_dependency(
                    entity_id,
                    attrs[self.ATTR_ENERGY_PARENT],
                    RelationType.ENERGY_PARENT,
                )
        
        self._cache_valid = True
        _LOGGER.debug(
            "[KnowledgeGraph] Loaded %d entities with dependencies",
            len(self._dependencies)
        )
    
    def _add_dependency(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        activation_mode: str = "auto",
        threshold: Optional[float] = None,
    ) -> None:
        """Add a dependency to the graph."""
        try:
            mode = ActivationMode(activation_mode.lower())
        except ValueError:
            mode = ActivationMode.AUTO
        
        dep = Dependency(
            source_entity=source,
            target_entity=target,
            relation_type=relation_type,
            activation_mode=mode,
            threshold=threshold,
        )
        
        if source not in self._dependencies:
            self._dependencies[source] = []
        self._dependencies[source].append(dep)
        
        # Build reverse index
        if target not in self._reverse_index:
            self._reverse_index[target] = []
        self._reverse_index[target].append(source)
    
    def get_dependencies(self, entity_id: str) -> List[Dependency]:
        """Get all dependencies for an entity."""
        if not self._cache_valid:
            self.refresh_graph()
        return self._dependencies.get(entity_id, [])
    
    def get_dependents(self, entity_id: str) -> List[str]:
        """Get entities that depend on this entity."""
        if not self._cache_valid:
            self.refresh_graph()
        return self._reverse_index.get(entity_id, [])
    
    def get_power_dependency(self, entity_id: str) -> Optional[str]:
        """Get the power source entity for a device."""
        deps = self.get_dependencies(entity_id)
        for dep in deps:
            if dep.relation_type == RelationType.POWERED_BY:
                return dep.target_entity
        return None
    
    def get_coupled_device(self, entity_id: str) -> Optional[str]:
        """Get the device this entity is coupled with."""
        deps = self.get_dependencies(entity_id)
        for dep in deps:
            if dep.relation_type == RelationType.COUPLED_WITH:
                return dep.target_entity
        return None
    
    def is_entity_usable(self, entity_id: str) -> Tuple[bool, Optional[str]]:
        """Check if entity can be used (dependencies are met).
        
        Returns:
            Tuple of (is_usable, reason_if_not)
        """
        deps = self.get_dependencies(entity_id)
        
        for dep in deps:
            target_state = self.hass.states.get(dep.target_entity)
            if not target_state:
                continue
            
            if dep.relation_type == RelationType.POWERED_BY:
                # Check if power source is on
                if target_state.state in ("off", "unavailable"):
                    return False, f"{dep.target_entity} ist aus"
            
            elif dep.relation_type == RelationType.COUPLED_WITH:
                # Check if coupled device is active
                if target_state.state in ("off", "unavailable", "idle", "standby"):
                    return False, f"{dep.target_entity} ist nicht aktiv"
        
        return True, None
    
    def resolve_for_action(
        self,
        entity_id: str,
        action: str,  # "turn_on", "turn_off", etc.
    ) -> DependencyResolution:
        """Resolve dependencies for an action on an entity.
        
        This is the main entry point for the conversation system.
        
        Args:
            entity_id: Target entity
            action: Intended action
            
        Returns:
            DependencyResolution with prerequisites and suggestions
        """
        if not self._cache_valid:
            self.refresh_graph()
        
        result = DependencyResolution(can_proceed=True)
        deps = self.get_dependencies(entity_id)
        
        for dep in deps:
            target_state = self.hass.states.get(dep.target_entity)
            if not target_state:
                continue
            
            # --- Power Dependencies ---
            if dep.relation_type == RelationType.POWERED_BY:
                if action in ("turn_on", "activate", "play"):
                    if target_state.state in ("off", "unavailable"):
                        friendly_name = target_state.attributes.get(
                            "friendly_name", dep.target_entity
                        )
                        
                        if dep.activation_mode == ActivationMode.AUTO:
                            result.prerequisites.append({
                                "entity_id": dep.target_entity,
                                "action": "turn_on",
                                "reason": f"Benötigt {friendly_name}",
                            })
                        elif dep.activation_mode == ActivationMode.WARN:
                            result.warnings.append(
                                f"{friendly_name} ist aus. Soll ich es einschalten?"
                            )
                            result.can_proceed = False
                        elif dep.activation_mode == ActivationMode.MANUAL:
                            result.warnings.append(
                                f"Hinweis: {friendly_name} ist aus."
                            )
            
            # --- Device Coupling ---
            elif dep.relation_type == RelationType.COUPLED_WITH:
                target_name = target_state.attributes.get(
                    "friendly_name", dep.target_entity
                )
                
                if action in ("turn_on", "activate"):
                    if target_state.state in ("off", "unavailable", "idle", "standby"):
                        if dep.activation_mode == ActivationMode.SYNC:
                            # Turn on together
                            result.prerequisites.append({
                                "entity_id": dep.target_entity,
                                "action": "turn_on",
                                "reason": f"Wird zusammen mit {target_name} aktiviert",
                            })
                        else:
                            # Just filter out - not useful
                            result.blocked_reason = f"{target_name} ist nicht aktiv"
                            result.can_proceed = False
                
                elif action in ("turn_off",) and dep.activation_mode == ActivationMode.SYNC:
                    # Turn off together
                    result.prerequisites.append({
                        "entity_id": dep.target_entity,
                        "action": "turn_off",
                        "reason": f"Wird zusammen mit {target_name} deaktiviert",
                    })
            
            # --- Lux-based Light Logic ---
            elif dep.relation_type == RelationType.LUX_SENSOR:
                if action in ("turn_on",) and entity_id.startswith("light."):
                    try:
                        current_lux = float(target_state.state)
                        threshold = dep.threshold or self.DEFAULT_LUX_THRESHOLD
                        
                        if current_lux > threshold:
                            result.suggestions.append(
                                f"Es ist bereits hell ({int(current_lux)} lux). "
                                "Möchtest du trotzdem das Licht einschalten?"
                            )
                    except (ValueError, TypeError):
                        pass
            
            # --- Associated Cover for Natural Light ---
            elif dep.relation_type == RelationType.ASSOCIATED_COVER:
                if action in ("turn_on",) and entity_id.startswith("light."):
                    cover_state = target_state
                    cover_name = cover_state.attributes.get(
                        "friendly_name", dep.target_entity
                    )
                    
                    # Check if cover is closed and it's daytime
                    if cover_state.state in ("closed",):
                        # Get lux sensor if available
                        lux_deps = [d for d in deps if d.relation_type == RelationType.LUX_SENSOR]
                        is_dark_outside = True
                        
                        if lux_deps:
                            lux_state = self.hass.states.get(lux_deps[0].target_entity)
                            if lux_state:
                                try:
                                    outside_lux = float(lux_state.state)
                                    is_dark_outside = outside_lux < 100
                                except (ValueError, TypeError):
                                    pass
                        
                        if not is_dark_outside:
                            result.suggestions.append(
                                f"{cover_name} ist geschlossen. "
                                "Soll ich stattdessen die Rollos öffnen?"
                            )
        
        return result
    
    def filter_candidates_by_usability(
        self, entity_ids: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter entity list to only include usable entities.
        
        Args:
            entity_ids: List of candidate entity IDs
            
        Returns:
            Tuple of (usable_entities, filtered_out_entities)
        """
        usable = []
        filtered = []
        
        for entity_id in entity_ids:
            is_usable, _ = self.is_entity_usable(entity_id)
            if is_usable:
                usable.append(entity_id)
            else:
                filtered.append(entity_id)
        
        if filtered:
            _LOGGER.debug(
                "[KnowledgeGraph] Filtered out %d unusable entities: %s",
                len(filtered), filtered
            )
        
        return usable, filtered
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge graph for debugging."""
        if not self._cache_valid:
            self.refresh_graph()
        
        return {
            "total_entities_with_deps": len(self._dependencies),
            "total_dependencies": sum(len(deps) for deps in self._dependencies.values()),
            "by_type": {
                rt.value: sum(
                    1 for deps in self._dependencies.values()
                    for d in deps if d.relation_type == rt
                )
                for rt in RelationType
            },
        }


# --- Module-level helper functions ---

_graph_cache: Dict[int, KnowledgeGraph] = {}


def get_knowledge_graph(hass: HomeAssistant) -> KnowledgeGraph:
    """Get or create the knowledge graph for a hass instance."""
    hass_id = id(hass)
    
    if hass_id not in _graph_cache:
        _graph_cache[hass_id] = KnowledgeGraph(hass)
    
    return _graph_cache[hass_id]


def resolve_dependencies(
    hass: HomeAssistant,
    entity_id: str,
    action: str,
) -> DependencyResolution:
    """Shortcut to resolve dependencies for an action."""
    graph = get_knowledge_graph(hass)
    return graph.resolve_for_action(entity_id, action)


def is_entity_usable(hass: HomeAssistant, entity_id: str) -> Tuple[bool, Optional[str]]:
    """Shortcut to check if entity is usable."""
    graph = get_knowledge_graph(hass)
    return graph.is_entity_usable(entity_id)


def filter_usable_entities(
    hass: HomeAssistant,
    entity_ids: List[str],
) -> List[str]:
    """Filter list to only usable entities."""
    graph = get_knowledge_graph(hass)
    usable, _ = graph.filter_candidates_by_usability(entity_ids)
    return usable
