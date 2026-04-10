import logging
from typing import List, Dict, Any

from .base import Capability
from ..utils.german_utils import COMPOUND_SEPARATOR, AREA_INDICATORS, FLOOR_KEYWORDS, LOCATION_INDICATORS

_LOGGER = logging.getLogger(__name__)


class AtomicCommandCapability(Capability):
    """Split compound commands into atomic actions.
    
    Handles commands connected with "und", "dann", or commas.
    Example: "Licht im Büro an und Rollo runter" -> ["Licht im Büro an", "Rollo runter"]
    """

    name = "atomic_command"
    description = "Split compound commands (e.g., 'A and B', 'A then B') into individual atomic actions using LLM reasoning. Ensures complex multi-device or multi-area requests are processed as separate steps."

    PROMPT = {
        "system": """You are a smart home command splitter.
Task: Split compound commands into atomic actions. Each action becomes one array element.

Rules:
1. Split on conjunctions ("und", "dann") and commas into separate array elements.
2. Each resulting command must be a single, self-contained action.
3. Preserve device names, area/floor references, and parameters in each split command.
4. Preserve time/duration constraints — never invent constraints not in the input.
5. If a conjunction connects different areas/floors with the same action, split per area.
6. If a single area has multiple actions, split per action.
7. Output the user's language (German).
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    def _has_multi_area_pattern(self, text: str) -> bool:
        """Check if text contains multi-area pattern that needs splitting.
        
        Detects patterns like:
        - "Küche und Büro" - two areas connected by und
        - "im Wohnzimmer und im Schlafzimmer" - explicit multi-area
        - "Erdgeschoss und Obergeschoss" - floor-based multi-area
        
        Does NOT trigger for:
        - "Licht und Rollladen" - different devices in same area
        - "auf 22 Grad und aus" - same area, different actions on same device
        """
        text_lower = text.lower()
        
        # Quick check: must contain compound separator
        if COMPOUND_SEPARATOR not in text_lower:
            return False
        
        # Split by separator and check if both sides look like area references
        parts = text_lower.split(COMPOUND_SEPARATOR)
        if len(parts) < 2:
            return False
        
        # Heuristic: Check for area indicator words on both sides
        # e.g., "im Wohnzimmer und im Schlafzimmer"
        has_area_left = any(ind in parts[0] for ind in AREA_INDICATORS)
        has_area_right = any(ind in parts[1] for ind in AREA_INDICATORS)
        
        if has_area_left and has_area_right:
            return True
        
        # Check for floor patterns: "Erdgeschoss und Obergeschoss"
        has_floor_pattern = (
            any(kw in parts[0] for kw in FLOOR_KEYWORDS) and
            any(kw in parts[1] for kw in FLOOR_KEYWORDS)
        )
        if has_floor_pattern:
            return True
        
        # Check for same action different areas pattern (simplified)
        # "Licht in der Küche an und im Flur aus"
        # Both parts mention a location indicator
        has_location_left = any(loc in parts[0] for loc in LOCATION_INDICATORS)
        has_location_right = any(loc in parts[1] for loc in LOCATION_INDICATORS)
        if has_location_left and has_location_right:
            return True
        
        return False

    async def run(self, user_input, **kwargs) -> List[str]:
        """Split compound commands if necessary."""
        text = user_input.text.strip()
        text_lower = text.lower()

        # Check for compound separators
        if any(sep in text_lower for sep in [COMPOUND_SEPARATOR, ",", "dann"]):
             _LOGGER.debug("[AtomicCommand] Compound separator detected, calling LLM")
             return await self._safe_prompt(
                self.PROMPT, {"user_input": text}, temperature=0.1
            )
        
        _LOGGER.debug("[AtomicCommand] No compound structure detected.")
        return [text]
