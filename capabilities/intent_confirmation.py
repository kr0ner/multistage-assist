import logging
from typing import Any, Dict, List, Optional
from .base import Capability

_LOGGER = logging.getLogger(__name__)


class IntentConfirmationCapability(Capability):
    """
    Produce a short natural-language confirmation of what will be executed
    for a given intent, entities, and parameters. Purely generative; no selection.
    """

    name = "intent_confirmation"
    description = "Create a concise German confirmation message for an intended action."

    PROMPT = {
        "system": """
You produce a short, natural confirmation of what will be executed. Write the final message in German (du-form).

## Input
- intent: internal intent name, e.g., "HassTurnOn", "HassTurnOff", "HassLightSet", "HassSetTemperature".
- entities: ordered list of objects:
  - "entity_id": string
  - "name": human-friendly display name (German)
- params: optional key/values (e.g., brightness, percentage, temperature, color, mode, duration, position, volume, scene, etc.)
- language: "de" by default. If another language is provided, write the final message in that language.
- style: "concise" by default – produce a single short but natural sentence.

## Rules
1) Always write the final confirmation in the requested language (default: German, du-form).
2) Refer to targets by their display names:
   - 1 target → mention it directly.
   - 2 targets → join with “und”.
   - >2 targets → comma-separated; last with “und”.
3) Derive meaning generically from `intent` and `params` (no domain hardcoding):
   - on/off → “einschalten”/“ausschalten”.
   - brightness/percentage/position/volume → “auf <value>%”.
   - temperature → “auf <value>°C”.
   - color → “auf <color>” (if present).
   - mode/scene → “auf <mode/scene>”.
   - duration/time → “für <duration>” or “um <time>” if applicable.
   - If nothing fits → brief generic confirmation (e.g., “Okay, ich führe das aus.”).
4) No explanations, no JSON in prose, just one sentence.
5) If `entities` is empty → state that there are no suitable targets.
6) Keep it brief, friendly, and natural.

## Examples (German outputs)
Input:
{
  "intent": "HassTurnOn",
  "entities": [{"entity_id":"light.bad_spiegel","name":"Spiegellicht"}],
  "params": {},
  "language": "de",
  "style": "concise"
}
Output:
{"message":"Alles klar, ich schalte Spiegellicht ein."}

Input:
{
  "intent": "HassLightSet",
  "entities": [{"entity_id":"light.kueche","name":"Küche"},{"entity_id":"light.kueche_spots","name":"Küche Spots"}],
  "params": {"brightness": 60, "color": "warmweiß"}
}
Output:
{"message":"Okay, ich stelle Küche und Küche Spots auf 60 % warmweiß."}

Input:
{
  "intent": "HassTurnOff",
  "entities": [{"entity_id":"switch.buero_lampe","name":"Bürolampe"}],
  "params": {}
}
Output:
{"message":"Alles klar, ich schalte Bürolampe aus."}

Input:
{
  "intent": "HassSetTemperature",
  "entities": [{"entity_id":"climate.wohnzimmer","name":"Thermostat Wohnzimmer"}],
  "params": {"temperature": 21.5}
}
Output:
{"message":"Alles klar, ich stelle Thermostat Wohnzimmer auf 21,5 °C."}

Input:
{
  "intent": "Any",
  "entities": [],
  "params": {}
}
Output:
{"message":"Hm, ich habe dafür gerade keine passenden Ziele."}
""",
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            }
        }
    }

    async def run(
        self,
        user_input,
        *,
        intent: str,
        entities: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
        language: str = "de",
        style: str = "concise",
        **_: Any,
    ) -> Dict[str, Any]:
        return await self._safe_prompt(
            self.PROMPT,
            {
                "intent": intent,
                "entities": entities,
                "params": params or {},
                "language": language,
                "style": style,
            },
            temperature=0.5,
        )
