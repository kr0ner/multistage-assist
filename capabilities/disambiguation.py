"""Disambiguation capability - asks user to clarify which device they meant.

This is a simple Python implementation - no LLM needed for formatting a question.
"""

import logging
from typing import Any, Dict

from .base import Capability
from ..constants.messages_de import DISAMBIGUATION_MESSAGES

_LOGGER = logging.getLogger(__name__)


class DisambiguationCapability(Capability):
    """Ask the user to clarify which device was meant."""

    name = "disambiguation"
    description = "Ask the user to clarify between multiple matched entities using natural language questions. Used when intent resolution is ambiguous."

    async def run(self, user_input, entities: Dict[str, str], **_: Any) -> Dict[str, Any]:
        """Generate disambiguation question from entity friendly names.
        
        Args:
            user_input: Original conversation input
            entities: Dict of entity_id -> friendly_name
            
        Returns:
            Dict with "message" key containing the question
        """
        names = list(entities.values())
        count = len(names)
        
        _LOGGER.debug("[Disambiguation] Generating question for %d candidates", count)
        
        if count == 0:
            return {"message": DISAMBIGUATION_MESSAGES["which_device"]}
        
        if count == 1:
            # Shouldn't happen, but handle it
            return {"message": DISAMBIGUATION_MESSAGES["mean_one"].format(name=names[0])}
        
        if count == 2:
            # "Meinst du X oder Y?"
            message = DISAMBIGUATION_MESSAGES["mean_two"].format(name1=names[0], name2=names[1])
        else:
            # "Welches meinst du: A, B, C oder D?"
            options = ", ".join(names[:-1]) + f" oder {names[-1]}"
            message = DISAMBIGUATION_MESSAGES["mean_multiple"].format(options=options)
        
        return {"message": message}
