import logging
from typing import Any, Dict, List
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class DisambiguationResolutionCapability(Capability):
    """Resolve which device(s) the user meant after clarification."""

    name = "disambiguation_resolution"
    description = "Interpret user clarification and resolve entities."

    PROMPT = {
    "system": """
You are resolving the user's follow-up answer after a clarification question about multiple devices.

## Input
- user_input: German response (e.g., "Spiegellicht", "erste", "zweite", "alle", "keine").
- input_entities: ordered list of objects, each with:
  - "entity_id": string
  - "name": friendly name (in German)

## Task
Map `user_input` to one or more `entity_id`s from `input_entities` and produce a short German confirmation (du-Form).

## Matching priority (apply in order)
1) Exact friendly-name match (case-sensitive). If found → choose it.
2) Normalized equality: compare lowercased, diacritic-insensitive, trimmed, with punctuation and extra spaces removed; treat "ue/ae/oe" as "ü/ä/ö".
3) Ordinal: "erste/zweite/dritte/vierte/…/letzte" and numeric forms ("1", "1.", "nr 1", "nummer 1") → pick the item at that 1-based position (or last for "letzte").
4) Plural all: "alle", "beide", "beiden" → return all entity_ids (if only one candidate exists, return that one).
5) None: "keine", "nichts", "nein" → return an empty list in `entities` and an acknowledgement.
6) If still unclear or confidence is low → return `{}` exactly.

## Confirmation message
- Always German du-Form.
- If one entity: mention its friendly name.
- If multiple: list friendly names comma-separated, short and natural.
- If `entities` is empty due to "keine"/"nichts": acknowledge doing nothing.

## Output format (STRICT)
Return ONLY a minified JSON object with exactly these keys in this order:
{"entities":[<strings>],"message":"<string>"}

- No markdown, no backticks, no extra text.
- Never return a bare array like ["light.kueche_spots"] — that is invalid.
- On failure/uncertainty, return `{}` exactly.

## Examples
Input:
{
  "user_input": "Spiegellicht",
  "input_entities": [
    {"entity_id": "light.badezimmer_spiegel", "name": "Badezimmer Spiegel"},
    {"entity_id": "light.badezimmer", "name": "Badezimmer"}
  ]
}
Output:
{"entities":["light.badezimmer_spiegel"],"message":"Okay, ich schalte das Spiegellicht ein."}

Input:
{
  "user_input": "erste",
  "input_entities": [
    {"entity_id": "light.kuche", "name": "Küche"},
    {"entity_id": "light.kuche_spots", "name": "Küche Spots"}
  ]
}
Output:
{"entities":["light.kuche"],"message":"Alles klar, ich nehme das erste: Küche."}

Input:
{
  "user_input": "alle",
  "input_entities": [
    {"entity_id": "light.kuche", "name": "Küche"},
    {"entity_id": "light.kuche_spots", "name": "Küche Spots"}
  ]
}
Output:
{"entities":["light.kuche","light.kuche_spots"],"message":"Alles klar, ich nehme beide: Küche und Küche Spots."}

Input:
{
  "user_input": "keine",
  "input_entities": [
    {"entity_id": "light.kuche", "name": "Küche"},
    {"entity_id": "light.kuche_spots", "name": "Küche Spots"}
  ]
}
Output:
{"entities":[],"message":"Alles klar - ich mache nichts."}
""",
        "schema": {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
                "message": {"type": "string"}
            }
        }
    }

    async def run(self, user_input, candidates: List[Dict[str, str]], **_: Any) -> Dict[str, Any]:
        return await self._safe_prompt(
            self.PROMPT,
            {"user_input": user_input.text, "input_entities": candidates},
            temperature=0.25,
        )
