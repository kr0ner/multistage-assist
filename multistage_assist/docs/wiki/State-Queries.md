# State Queries

Ask about the current state of devices and get informative responses.

## Light Queries

| German Command | What Happens |
|----------------|--------------|
| "Welche Lichter sind an?" | Lists all lights that are on |
| "Sind Lampen an?" | Reports if any lights are on |
| "Ist das Licht im Bad an?" | Reports bathroom light state |
| "Welche Lampen sind aus?" | Lists lights that are off |

### Response Examples

```
User: "Welche Lichter sind an?"
Assistant: "Folgende Lichter sind an: Küche, Wohnzimmer, Flur EG."
```

```
User: "Sind Lampen an?"
Assistant: "Ja, 3 Lichter sind an."
```

```
User: "Welche Lichter sind an?"
(when none are on)
Assistant: "Keine Geräte sind on."
```

## Temperature Queries

| German Command | What Happens |
|----------------|--------------|
| "Wie warm ist es?" | Reports current temperature |
| "Wie ist die Temperatur?" | Reports temperature sensor |
| "Wie viel Grad sind es?" | Reports temperature |
| "Wie ist die Temperatur im Wohnzimmer?" | Reports room temperature |

## Cover Queries

| German Command | What Happens |
|----------------|--------------|
| "Ist das Rollo oben?" | Reports cover state |
| "Sind die Rolläden geschlossen?" | Reports cover positions |

## General State Queries

Use `HassGetState` intent with:

| Slot | Description |
|------|-------------|
| `domain` | Entity domain (light, cover, etc.) |
| `state` | Filter by state (on, off, open, closed) |
| `area` | Filter by area |
| `floor` | Filter by floor |

## Query Processing

1. **Entity Resolution** - Finds all matching entities
2. **State Filtering** - Filters by requested state
3. **Response Generation** - Creates natural language response

## Yes/No Questions

Simple yes/no questions about state:

| German Command | Response Type |
|----------------|---------------|
| "Ist das Licht an?" | Yes/No |
| "Sind Lichter an?" | Yes/No with count |
| "Ist die Tür offen?" | Yes/No |

## Multi-Entity Responses

When querying multiple entities:

```
User: "Welche Lichter sind an?"
→ Finds all light entities
→ Filters by state = "on"
→ Lists names or counts
```

## Supported Query Domains

| Domain | Example Query |
|--------|---------------|
| `light` | "Welche Lichter sind an?" |
| `cover` | "Sind Rolläden offen?" |
| `switch` | "Welche Schalter sind an?" |
| `sensor` | "Wie warm ist es?" |
| `climate` | "Wie ist die Temperatur?" |

## Example Flows

### Light State Query
```
User: "Welche Lichter sind an?"
→ Intent: HassGetState
→ Domain: light
→ State: on
→ Finds: light.kuche, light.wohnzimmer
Assistant: "Küche und Wohnzimmer."
```

### Temperature Query
```
User: "Wie warm ist es im Bad?"
→ Intent: HassGetState
→ Domain: sensor
→ Area: Bad
Assistant: "Im Bad sind es 22 Grad."
```

### Empty Result
```
User: "Welche Lichter sind an?"
(when all lights are off)
Assistant: "Keine Geräte sind an."
```
