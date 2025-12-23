# Delayed Controls

Schedule device actions for a future time using natural language.

## Delay-Based Commands

| Command | What Happens |
|---------|--------------|
| "Schalte in 10 Minuten das Licht aus" | Turns off light after 10 minutes |
| "Mach in einer Stunde die Heizung aus" | Turns off heating after 1 hour |
| "Schließe in 5 Minuten die Rollläden" | Closes covers after 5 minutes |

## Time-Based Commands

| Command | What Happens |
|---------|--------------|
| "Mach um 15:30 das Licht an" | Turns on light at 15:30 (today or tomorrow) |
| "Schalte um 22 Uhr die Heizung aus" | Turns off heating at 22:00 |
| "Öffne um 7 Uhr die Rollläden" | Opens covers at 7:00 |

**Note**: If the specified time has already passed today, the action is scheduled for tomorrow.

## Duration Formats

- X Minuten / X Minute
- X Sekunden / X Sekunde
- X Stunden / X Stunde

## Time Formats

- HH:MM (e.g., "15:30")
- HH Uhr (e.g., "15 Uhr")
- um HH:MM / um HH Uhr

## Supported Domains

- light
- cover
- switch
- fan
- automation

## How It Works

1. **Parse** - Extract delay duration or target time
2. **Calculate** - Convert to delay in minutes/seconds
3. **Schedule** - Call `delay_action` script
4. **Execute** - Script waits, then performs action

## Required Script

The `delay_action` script must be installed:

```bash
cp multistage_assist/scripts/delay_action.yaml /config/scripts/
```

Then reload scripts: **Developer Tools → YAML → Reload Scripts**

## Script Parameters

| Parameter | Description |
|-----------|-------------|
| `target_entity` | Entity to control |
| `action` | "on" or "off" |
| `value` | Numeric value (brightness, position) |
| `minutes` | Delay minutes |
| `seconds` | Delay seconds |

## Difference from Temporary Controls

| Feature | Delayed Controls | Temporary Controls |
|---------|-----------------|-------------------|
| Keyword | "in X Minuten" | "für X Minuten" |
| Behavior | Wait, then act | Act, wait, restore |
| Script | `delay_action` | `timebox_entity_state` |

## Cache Behavior

Delayed control commands are **never cached** because:
- Each delay is unique (time-sensitive)
- The same phrase at different times should produce different delays;
