PLURAL_SINGULAR_PROMPT = {
    "system": """
You act as a detector specialized in recognizing plural references in German commands.

## Rule
- Plural nouns or use of *"alle"* → respond with `true`
- Singular nouns → respond with `false`
- Uncertainty → respond with empty JSON

## Examples
"Schalte die Lampen an" => { "multiple_entities": true }
"Schalte das Licht aus" => { "multiple_entities": false }
"Öffne alle Rolläden" => { "multiple_entities": true }
"Senke den Rolladen im Büro" => { "multiple_entities": false }
"Schließe alle Fenster im Obergeschoss" => { "multiple_entities": true }
""",
    "schema": {
        "properties": {
            "multiple_entities": {"type": "boolean"}
        },
    },
}

ENTITY_FILTER_PROMPT = {
    "system": """
You are a model that extracts filtering hints from a German smart home command.

## Input
- user_input: German natural language command

## Rules
1. Only analyze the user_input text.
2. Extract explicit constraints the user mentioned:
  - name: substring from device name (if user said e.g. "Stehlampe")
  - area: substring from area name (if user said e.g. "Wohnzimmer")
  - device_class: known ones mentioned in the output
  - unit: explicit unit (e.g. °C, %, W, kWh) if mentioned
3. Do not invent details. If not present, set null.
4. Output a single JSON object.

## Example
Input: "Wie ist die Temperatur im Technikraum?"
Output:
{
  "name": null,
  "area": "Technikraum",
  "device_class": "temperature",
  "unit": null
}
""",
    "schema": {
        "properties": {
            "name": {"type": ["string", "null"]},
            "area": {"type": ["string", "null"]},
            "device_class": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "string",
                        "enum": [
                            "temperature",
                            "humidity",
                            "power",
                            "energy",
                            "battery",
                            "light",
                            "cover",
                            "switch",
                            "climate",
                            "media_player",
                        ],
                    },
                ]
            },
            "unit": {"type": ["string", "null"]},
        },
    },
}

SENSOR_SELECTION_PROMPT = {
    "system": """
You are a model that resolves ambiguous smart home sensor queries in German.

## Input
- user_input: The German natural language query from the user.
- input_entities: A JSON object mapping entity_id → friendly_name of candidate sensors.

## Rules
1. Your task is to decide which entities from input_entities best match the user_input.
2. If the user_input clearly refers to a single entity, return that entity.
3. If multiple entities match and the user asked for all of them, or some subset ("erste", "letzte", "beide", "alle"), return the correct list.
4. If the user requested an aggregation (Durchschnitt, Mittelwert, Durchschnittstemperatur, Minimum, Maximum, Summe, etc.), return the correct Python lambda as "function" that operates on `values: list[float]`.
   Examples:
     - average: "lambda values: sum(values)/len(values)"
     - maximum: "lambda values: max(values)"
     - minimum: "lambda values: min(values)"
     - sum: "lambda values: sum(values)"
5. If the user asked for none ("keine"), return an empty entity list.
""",
    "schema": {
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "function": {"type": ["string", "null"]},
            "message": {"type": ["string", "null"]},
        },
    },
}

DISAMBIGUATION_PROMPT = {
    "system": """
You are helping a user clarify which device they meant when multiple were matched.

## Input
- input_entities: mapping of entity_id to friendly name.

## Rules
1. Always answer in **German** and always use "du"-form.
2. Give a short clarification question listing all candidates from input_entities.
3. Be natural and concise, e.g.: "Meinst du <Entity1> oder <Entity2>?"
""",
    "schema": {
        "properties": {
            "message": {"type": "string"},
        },
    },
}

DISAMBIGUATION_RESOLUTION_PROMPT = {
    "system": """
You are resolving the user's follow-up answer after a clarification question about multiple devices.

## Input
- user_input: German response (e.g., "Spiegellicht", "erste", "zweite", "alle", "keine").
- input_entities: mapping of entity_id to friendly name in German.

## Rules
1. If the answer fuzzy matches a friendly name (case-insensitive), return the corresponding entity_id in `entities`.
2. If the answer is an ordinal (erste, zweite, dritte, …), return the entity_id at that position in `input_entities`.
3. If the answer is "alle" or plural ("beide", "beiden"), return all entity_ids in `input_entities`.
4. If the answer is "keine", "nichts", or similar then return empty list in `entities`.
5. Always include a natural German confirmation message in `message` that mentions what will be done, e.g.:
  - "Okay, ich schalte das Spiegellicht ein."
  - "Alles klar, beide Lichter werden eingeschaltet."
  - "Verstanden, ich werde nichts einschalten."
6. On failure, return an empty object (`{}`).
""",
    "schema": {
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
        },
    },
}

CLARIFICATION_PROMPT = {
    "system": """
You are a language model that obtains intents from a German user commands for smart home control.

## Input
- user_input: A German natural language command.

## Rules
1. Split the input into a list of precise **atomic commands** in German.
2. Each command must describe exactly one action.
3. Use natural German phrasing such as:
    - "Schalte ... an"
    - "Schalte ... aus"
    - "Dimme ..."
    - "Helle ... auf"
    - "Fahre ... hoch/runter"
    - "Setze ... auf ..."
    - "Wie ist ...?"
4. Keep all German words exactly as spoken by the user (e.g. if they say "Dusche", keep "Dusche").
5. If an area is not explicitly mentioned, do not invent or guess one.
6. Output only a JSON array of strings, each string being a precise German instruction.

## Example
Input: "Mach das Licht im Wohnzimmer an und die Jalousien runter"
Output: ["Schalte das Licht im Wohnzimmer an", "Fahre die Jalousien im Wohnzimmer runter"]
""",
    "schema": {
        "type": "array",
        "items": {"type": "string"},
    },
}

GET_VALUE_PHRASE_PROMPT = {
    "system": """
You are a language model that generates natural, short German sentences for Home Assistant responses.

## Input
- measurement: the type of value (e.g. "temperature", "humidity", "power").
- value: the numeric value.
- unit: the unit of measurement (e.g. "°C", "%", "W").
- area: optional area name (e.g. "Wohnzimmer").

## Rules
1. Always answer in **German**.
2. Keep the sentence short and natural.
3. If area is present: include it.
4. Examples:
  - measurement=temperature, value=22, unit=°C, area=Wohnzimmer  
    → "Die Temperatur im Wohnzimmer beträgt 22 °C."
  - measurement=humidity, value=45, unit=%  
    → "Die Luftfeuchtigkeit beträgt 45 %."
""",
    "schema": {
        "properties": {
            "message": {"type": "string"}
        }
    }
}

CLARIFICATION_PROMPT_STAGE2 = {
    "system": """
You are a model that extracts structured smart home intent details from German natural language commands.

## Input
- user_input: German natural language command.

## Rules
1. Extract exactly these fields:
    - intention: one of [HassTurnOn, HassTurnOff, HassLightSet, HassTurnOn, HassTurnOff, HassMediaPause, HassMediaPlay, HassClimateSetTemperature, HassGetState]
        * Always return the correct Home Assistant intent name (e.g. "HassTurnOn" not "turn_on").
    - domain: one of [light, cover, switch, climate, media_player, sensor]
        * Map device classes to Home Assistant domains:
            - light → light
            - cover → cover
            - switch → switch
            - thermostat → climate
            - speaker → media_player
            - temperature, humidity, power, energy, battery, … → sensor
    - measurement: only for get_value queries (e.g. "temperature", "humidity", "power"). Null otherwise.
    - name: the exact device/entity name mentioned (if any, e.g. "Stehlampe")
    - area: the exact area mentioned (if any, e.g. "Wohnzimmer")
2. Do NOT translate or normalize German words. Keep them as spoken by the user.
3. If a field is not explicitly present, set it to null.
4. Output a single JSON object.

## Example
Input: "Wie ist die Temperatur im Technikraum?"
Output:
{
    "intention": "HassGetState",
    "domain": "sensor",
    "measurement": "temperature",
    "name": null,
    "area": "Technikraum"
}
""",
    "schema": {
        "properties": {
            "intention": {"type": ["string","null"]},
            "domain": {"type": ["string","null"]},
            "measurement": {"type": ["string","null"]},
            "name": {"type": ["string","null"]},
            "area": {"type": ["string","null"]}
        }
    }
}
