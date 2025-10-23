import logging
from typing import Any, Dict, List, Union
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class DisambiguationSelectCapability(Capability):
    """
    Map the user's follow-up answer to one or more entity_ids from the given candidates.
    No message generation here.
    """

    name = "disambiguation_select"
    description = "Select entity_ids from candidates based on user_input."

    PROMPT = {
        "system": """
You select which candidates the user meant. Do not write explanations.

## Input
- user_input: German response (e.g., "Spiegellicht", "erste", "zweite", "alle", "keine").
- input_entities: ordered list of { "entity_id": string, "name": string, "ordinal": integer }.

## Rules
1. Ordinals: The field "ordinal" gives the numeric order for each candidate.
   - "erste" → ordinal = 1
   - "zweite" → ordinal = 2
   - "dritte" → ordinal = 3
   - "vierte" → ordinal = 4
   - "letzte" → highest ordinal in input_entities
   - Numeric forms ("1", "1.", "nr 1", "nummer 1") map to ordinal 1
2. Friendly name fuzzy matching:
   - Normalize lowercased names (ignore accents, trim spaces/punctuation, treat ue/oe/ae ~ ü/ö/ä)
3. "alle" → return all entity_ids.
4. "beide", "beiden" → return all entity_ids if length of input is two.
5. "keine", "nichts", "nein" → return an empty array.
6. If uncertain or ambiguous → return an empty array.

## Output (STRICT)
Return ONLY a minified JSON array of strings (entity_ids). No other text.
Example valid: ["light.badezimmer_spiegel"]
Example valid (none): []
""",
        "schema": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    async def run(self, user_input, candidates: List[Dict[str, str]], **_: Any) -> List[str]:
        raw: Union[List[str], Dict[str, Any], None] = await self._safe_prompt(
            self.PROMPT,
            {"user_input": user_input.text, "input_entities": candidates},
            temperature=0.0,
        )
        # Be minimal but safe: accept either array or {"entities":[...]}
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, str)]
        if isinstance(raw, dict) and isinstance(raw.get("entities"), list):
            return [x for x in raw["entities"] if isinstance(x, str)]
        return []
