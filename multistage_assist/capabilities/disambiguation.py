import logging
from typing import Any, Dict
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class DisambiguationCapability(Capability):
    """Ask the user to clarify which device was meant."""

    name = "disambiguation"
    description = "Ask the user to clarify between multiple matched entities."

    PROMPT = {
    "system": """
You are helping a user clarify which device they meant when multiple were matched.

## Input
- input_entities: mapping of entity_id to friendly name.

## Rules
1. Always answer in German and always use "du"-form.
2. Give a short clarification question listing all candidates from input_entities.
3. Be natural and concise, e.g.: "Meinst du Entity1 oder Entity2?

## Examples
input_entities -> { "light.livingroom": "Wohnzimmerlicht", "switch.office_lamp": "Bürolampe" }
message -> "Meinst du Wohnzimmerlicht oder Bürolampe?"

input_entities -> { "climate.bath": "Thermostat Bad", "climate.kitchen": "Thermostat Küche", "climate.bedroom": "Thermostat Schlafzimmer" }
message -> "Welches meinst du: Thermostat Bad, Thermostat Küche oder Thermostat Schlafzimmer?"

input_entities -> { "media.apple_tv_living": "Apple TV - Wohnzimmer", "media.apple_tv_bed": "Apple TV - Schlafzimmer" }
message -> "Meinst du Apple TV - Wohnzimmer oder Apple TV - Schlafzimmer?"

input_entities -> { "light.desk": "Schreibtischlampe", "light.ceiling_office": "Deckenlicht Büro", "light.shelf": "Lichtleiste Regal", "light.monitor": "Monitorlicht" }
message -> "Welches davon meinst du: Schreibtischlampe, Deckenlicht Büro, Lichtleiste Regal oder Monitorlicht?"
""",
        "schema": {
            "properties": {
                "message": {"type": "string"},
            },
        },
    }

    async def run(self, user_input, entities: Dict[str, str], **_: Any) -> Dict[str, Any]:
        ordered = [{"entity_id": eid, "name": name} for eid, name in entities.items()]
        _LOGGER.debug("[Disambiguation] Prompting user to clarify between %d options", len(ordered))
        return await self._safe_prompt(self.PROMPT, {"input_entities": ordered})
