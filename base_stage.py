"""Base stage orchestrator for the multi-stage pipeline.

Each stage inherits from BaseStage and implements the process() method
to handle user input and either resolve intent/entities or escalate.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from .capabilities.base import Capability
from .stage_result import StageResult

_LOGGER = logging.getLogger(__name__)


class BaseStage(ABC):
    """Base stage orchestrator managing a collection of capabilities.
    
    Each stage processes user input and returns a StageResult indicating:
    - success: Intent + entities resolved, ready for execution
    - escalate: Need more sophisticated processing, pass to next stage
    - escalate_chat: User wants to chat, pass to Stage3 in chat-only mode
    - error: Unrecoverable error with response
    """

    name = "base"
    capabilities: List[Type[Capability]] = []

    def __init__(self, hass: Any, config: Dict[str, Any]) -> None:
        self.hass = hass
        self.config = config
        self.capabilities_map: Dict[str, Capability] = {
            cap.name: cap(hass, config) for cap in self.capabilities
        }

    def has(self, name: str) -> bool:
        """Check if a capability is available."""
        return name in self.capabilities_map

    def get(self, name: str) -> Capability:
        """Get a capability by name."""
        if name not in self.capabilities_map:
            raise KeyError(f"Capability '{name}' not found in stage {self.name}")
        return self.capabilities_map[name]

    async def use(self, name: str, user_input, **kwargs) -> Any:
        """Use a capability and return its result."""
        cap = self.get(name)
        _LOGGER.debug(
            "[%s] Using capability '%s' with kwargs=%s",
            self.name,
            name,
            list(kwargs.keys()),
        )
        result = await cap.run(user_input, **kwargs)
        _LOGGER.debug("[%s] Capability '%s' returned: %s", self.name, name, result)
        return result

    @abstractmethod
    async def process(self, user_input, context: Dict[str, Any] = None) -> StageResult:
        """Process user input and return a stage result.
        
        Args:
            user_input: ConversationInput from Home Assistant
            context: Optional context from previous stage (enriched data)
            
        Returns:
            StageResult with status indicating next action
        """
        pass
