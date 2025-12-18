# Knowledge Graph Configuration

MultiStage Assist includes a **Knowledge Graph** system that models functional dependencies between Home Assistant entities. This enables intelligent behaviors like:

- Automatically turning on power switches before a device can be used
- Filtering out devices that aren't currently usable (e.g., Ambilight when TV is off)
- Suggesting alternatives (e.g., opening blinds instead of turning on lights during the day)

## Why a Knowledge Graph?

Home Assistant handles **State** and **Location** well, but lacks a native model for:
- **Power Dependencies**: Kitchen radio needs kitchen power socket
- **Device Coupling**: Ambilight is only useful when TV is on
- **Contextual Logic**: Light decisions based on current brightness

## Configuration

Define relationships using custom attributes in `customize.yaml`:

```yaml
homeassistant:
  customize:
    # Power Dependency
    # The kitchen radio needs the kitchen main power switch to be on
    media_player.kitchen_radio:
      powered_by: switch.kitchen_main_power
      activation_mode: auto  # auto, warn, sync, manual

    # Device Coupling
    # Ambilight is only useful when the TV is on
    light.living_room_ambilight:
      coupled_with: media_player.living_room_tv
      activation_mode: sync  # Turn on/off together

    # Light Context
    # Room light can consider the lux sensor and suggest opening covers
    light.living_room_main:
      lux_sensor: sensor.living_room_lux
      min_lux_threshold: 200
      associated_cover: cover.living_room_blinds

    # Energy Hierarchy
    # For tracking consumption chains
    sensor.living_room_power:
      energy_parent: sensor.house_total_power
```

## Relationship Types

### `powered_by`
Defines a power dependency between devices.

```yaml
media_player.kitchen_radio:
  powered_by: switch.kitchen_main_power
```

**Behavior**: When user asks to turn on the radio, the system will:
1. Check if the power switch is on
2. If off + `auto` mode: Turn on the switch first
3. If off + `warn` mode: Ask user if they want to turn on the switch
4. Then proceed with the original request

### `coupled_with`
Defines a device coupling where one device is only useful when another is active.

```yaml
light.living_room_ambilight:
  coupled_with: media_player.living_room_tv
  activation_mode: sync
```

**Behavior**: 
- Entity is filtered from candidates if coupled device is off
- With `sync` mode: Both devices turn on/off together
- With `warn` mode: User is informed that coupled device is off

### `lux_sensor`
Associates a light sensor with a light entity for contextual decisions.

```yaml
light.living_room_main:
  lux_sensor: sensor.living_room_lux
  min_lux_threshold: 200
```

**Behavior**: When user asks to turn on the light:
- If lux > threshold: Suggest it's already bright enough
- If lux < threshold: Proceed normally

### `associated_cover`
Links a cover to a light for natural light suggestions.

```yaml
light.living_room_main:
  associated_cover: cover.living_room_blinds
```

**Behavior**: When user asks to turn on the light during daytime:
- If cover is closed: Suggest opening the cover instead of turning on the light

### `activation_mode`
Controls how dependencies are handled:

| Mode | Behavior |
|------|----------|
| `auto` | Automatically enable dependency (default for power) |
| `warn` | Warn user and ask for confirmation |
| `sync` | Keep devices in sync (turn on/off together) |
| `manual` | Just inform, don't suggest or auto-enable |

## Example Use Cases

### 1. Kitchen Radio with Power Socket

```yaml
media_player.kitchen_radio:
  powered_by: switch.kitchen_main_power
  activation_mode: auto
```

User: "Schalte das Küchenradio ein"
System: 
1. Checks: switch.kitchen_main_power is OFF
2. Automatically turns ON switch.kitchen_main_power
3. Turns ON media_player.kitchen_radio
Response: "Ich habe den Küchenstrom und das Radio eingeschaltet."

### 2. Ambilight with TV

```yaml
light.living_room_ambilight:
  coupled_with: media_player.living_room_tv
  activation_mode: sync
```

User: "Mach das Ambilight an"
- If TV is ON: Turns on Ambilight
- If TV is OFF + sync mode: Turns on both TV and Ambilight
- If TV is OFF + warn mode: "Der Fernseher ist aus. Soll ich beides einschalten?"

### 3. Smart Light Decision

```yaml
light.living_room_main:
  lux_sensor: sensor.living_room_lux
  min_lux_threshold: 200
  associated_cover: cover.living_room_blinds
```

User: "Mach Licht im Wohnzimmer an" (during daytime, curtains closed)
System: "Die Rollos sind geschlossen. Soll ich stattdessen die Rollos öffnen?"

### 4. Filtering Unusable Devices

When asking "Welche Lichter kann ich im Wohnzimmer steuern?":
- Devices with unmet `powered_by` dependencies are marked as "(benötigt Strom)"
- Devices with unmet `coupled_with` dependencies are filtered out entirely

## Integration with MultiStage Assist

The Knowledge Graph integrates automatically with the conversation flow:

1. **Entity Resolution**: Unusable entities can be filtered from candidate lists
2. **Intent Execution**: Prerequisites are resolved before main action
3. **Confirmation**: User is informed about any automatic actions taken

### Using in Code

```python
from ..utils.knowledge_graph import get_knowledge_graph, resolve_dependencies

# Get the graph
graph = get_knowledge_graph(hass)

# Resolve dependencies before an action
resolution = graph.resolve_for_action(entity_id, "turn_on")

if resolution.prerequisites:
    for prereq in resolution.prerequisites:
        await hass.services.async_call(
            prereq["entity_id"].split(".")[0],
            prereq["action"],
            {"entity_id": prereq["entity_id"]},
        )

if resolution.suggestions:
    # Show alternative suggestions to user
    ...

if resolution.can_proceed:
    # Execute main action
    ...
```

## Debugging

View the graph structure:

```python
graph = get_knowledge_graph(hass)
summary = graph.get_graph_summary()
# {
#   "total_entities_with_deps": 5,
#   "total_dependencies": 7,
#   "by_type": {"powered_by": 2, "coupled_with": 1, ...}
# }
```

## Best Practices

1. **Start Simple**: Add power dependencies first, then expand
2. **Use `warn` Mode**: For uncertain relationships, let user confirm
3. **Test Incrementally**: Add one relationship at a time
4. **Document Your Graph**: Keep a diagram of major dependencies
5. **Consider Cycles**: Don't create circular dependencies

## Limitations

- The graph is refreshed on first access; use `graph.refresh_graph()` to force update
- Circular dependencies are not detected (avoid them in configuration)
- Energy hierarchy is informational only (no automatic actions)
