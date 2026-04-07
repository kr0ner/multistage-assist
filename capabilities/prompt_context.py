"""Capability for building LLM prompt context snapshots.

Provides a unified way to describe the Home Assistant environment (areas, floors, entities)
to LLMs, with support for intent-specific filtering.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from homeassistant.helpers import (
    area_registry as ar,
    floor_registry as fr,
    entity_registry as er,
)
from .base import Capability
from ..constants.messages_de import PROMPT_CONTEXT_MESSAGES, ERROR_MESSAGES

_LOGGER = logging.getLogger(__name__)


class PromptContextBuilderCapability(Capability):
    """
    Builds a structured 'Environment Snapshot' for LLM prompts.
    
    Supports:
    - Full house overview (for Stage 3)
    - Filtered/Hinted context (for Stage 2)
    - Consistent formatting across stages
    """

    name = "prompt_context"
    description = "Generate a structured overview of the home environment for LLMs."

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.knowledge_graph = None

    def set_knowledge_graph(self, kg_cap):
        """Inject knowledge graph capability."""
        self.knowledge_graph = kg_cap

    async def get_context(self, hints: List[str] = None) -> str:
        """Get a formatted string representing the home environment.
        
        Args:
            hints: List of keywords/names to filter the context (e.g. ['Licht', 'Küche'])
            
        Returns:
            Formatted context string
        """
        area_registry = ar.async_get(self.hass)
        floor_registry = fr.async_get(self.hass)
        entity_registry = er.async_get(self.hass)

        areas = list(area_registry.async_list_areas())
        floors = list(floor_registry.async_list_floors())

        # 1. Filter areas/floors if hints are provided
        if hints:
            filtered_areas = []
            filtered_floors = []
            
            # Simple keyword matching for hints
            norm_hints = [h.lower() for h in hints if h]
            
            for area in areas:
                if any(h in area.name.lower() or any(h in a.lower() for a in (area.aliases or [])) for h in norm_hints):
                    filtered_areas.append(area)
            
            for floor in floors:
                if any(h in floor.name.lower() or any(h in a.lower() for a in (floor.aliases or [])) for h in norm_hints):
                    filtered_floors.append(floor)
                    # Also include all areas on this floor
                    for area in areas:
                        if area.floor_id == floor.floor_id and area not in filtered_areas:
                            filtered_areas.append(area)
            
            # If nothing matched the hints, don't filter (fallback to full or minimal?)
            # Actually, standard behavior: if hints exist but no match, return minimal house info
            if filtered_areas or filtered_floors:
                areas = filtered_areas
                floors = filtered_floors

        # 2. Build Area/Floor String
        none_str = ERROR_MESSAGES.get("none", "Keine")
        area_names = sorted([a.name for a in areas if a.name])
        floor_names = sorted([f.name for f in floors if f.name])

        lines = [
            PROMPT_CONTEXT_MESSAGES["available_rooms"].format(
                rooms=', '.join(area_names) if area_names else none_str
            ),
            PROMPT_CONTEXT_MESSAGES["available_floors"].format(
                floors=', '.join(floor_names) if floor_names else none_str
            ),
        ]

        # 3. Add Personal Information from Knowledge Graph
        if self.knowledge_graph:
            personal_data = await self.knowledge_graph.get_all_personal_data()
            if personal_data:
                lines.append(PROMPT_CONTEXT_MESSAGES["personal_info_header"])
                for k, v in personal_data.items():
                    lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    async def run(self, user_input, **kwargs) -> Dict[str, Any]:
        """Capability run method (standard interface)."""
        hints = kwargs.get("hints")
        context = await self.get_context(hints=hints)
        return {"context": context}
