# Temporary Controls

Execute actions for a specified duration, then automatically restore the previous state.

## Supported Domains

- light
- cover
- switch
- fan
- automation

## Examples

| Command | What Happens |
|---------|--------------|
| "Licht für 5 Minuten an" | On, then restores after 5 min |
| "Licht für 10 Minuten aus" | Off, then restores after 10 min |
| "Rollo auf 50% für 10 Minuten" | Sets to 50%, then restores |
| "Türklingel für 30 Minuten aus" | Disables automation for 30 min |

## Duration Formats

- X Minuten / X Minute
- X Sekunden / X Sekunde
- X Stunden / X Stunde

## How It Works

1. **Snapshot** - Saves current entity state to a scene
2. **Execute** - Performs the requested action
3. **Wait** - Delays for specified duration
4. **Restore** - Activates the saved scene

Uses the `timebox_entity_state` script.

## Script Parameters

| Parameter | Description |
|-----------|-------------|
| target_entity | Entity to control |
| action | "on" or "off" |
| value | Numeric value (brightness, position) |
| minutes | Duration minutes |
| seconds | Duration seconds |
| scene_id | Lowercase ID for snapshot |
