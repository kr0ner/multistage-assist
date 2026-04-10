# MultiStage Assist

**MultiStage Assist** is a German-language voice assistant integration for Home Assistant. It uses a multi-stage processing pipeline with LLM capabilities to understand natural language commands.

## Features

- 🇩🇪 **Native German** - Optimized for German commands
- 🏠 **Entity Resolution** - Fuzzy matching for areas and devices  
- ⏱️ **Temporary Controls** - "für 5 Minuten" with auto-restore
- 📅 **Calendar** - Create events with natural language
- ⏲️ **Timers** - Timer creation with device notifications
- 🧹 **Vacuum** - Room-specific cleaning with mode detection

## Supported Domains

| Domain | Intents | Notes |
|--------|---------|-------|
| `light` | TurnOn, TurnOff, LightSet, GetState, TemporaryControl | Brightness step up/down |
| `cover` | TurnOn, TurnOff, SetPosition, GetState, TemporaryControl | |
| `switch` | TurnOn, TurnOff, GetState, TemporaryControl | |
| `fan` | TurnOn, TurnOff, GetState, TemporaryControl | |
| `climate` | ClimateSetTemperature, TurnOn, TurnOff, GetState | |
| `media_player` | TurnOn, TurnOff, GetState | |
| `sensor` | GetState | Temperature, humidity queries |
| `timer` | TimerSet | Duration + optional device |
| `vacuum` | VacuumStart | Room + mode (vacuum/mop) |
| `calendar` | CalendarCreate | Multi-turn event creation |
| `automation` | TurnOn, TurnOff, TemporaryControl | e.g., doorbells |

---

# Command Examples

## Lights

| Command | What Happens |
|---------|--------------|
| "Licht im Bad an" | Turns on bathroom lights |
| "Licht auf 50%" | Sets brightness to 50% |
| "Licht heller" | Increases brightness by 35% of current |
| "Licht dunkler" | Decreases brightness by 35% of current |
| "Es ist zu dunkel" | → "Mache das Licht heller" (via clarification) |
| "Licht für 5 Minuten an" | Turns on, restores after 5 min |

### Brightness Step Logic
- **Relative changes**: 35% of current value (not absolute)
- **Minimum step**: 10%
- **From off (0%)**: Turns on to 30%

## Covers

| Command | What Happens |
|---------|--------------|
| "Rollo runter" | Closes cover |
| "Rollo hoch" | Opens cover |
| "Rollo auf 50%" | Sets position to 50% |
| "Rollo für 10 Minuten auf 50%" | Sets to 50%, restores after 10 min |

## Climate

| Command | What Happens |
|---------|--------------|
| "Heizung auf 22 Grad" | Sets thermostat to 22°C |
| "Heizung im Wohnzimmer auf 21 Grad" | Sets specific room |

## Vacuum

| Command | Mode | Area |
|---------|------|------|
| "Sauge die Küche" | vacuum | Küche |
| "Wische den Keller" | mop | Keller |
| "Staubsauge das Erdgeschoss" | vacuum | Erdgeschoss |

**Mode detection:**
- `vacuum`: saugen, sauge, staubsaugen
- `mop`: wischen, wische, nass, feucht, moppen

## Timers

| Command | What Happens |
|---------|--------------|
| "Timer für 10 Minuten" | Creates 10-min timer |
| "Timer für 5 Minuten Eier" | Named timer "Eier" |

## Calendar

Multi-turn flow for event creation:

```
User: "Erstelle einen Termin morgen um 15 Uhr"
Assistant: "Wie soll der Termin heißen?"
User: "Zahnarzt"
Assistant: [shows preview]
User: "Ja"
Assistant: "Termin wurde erstellt."
```

**Relative dates supported:**
- heute, morgen, übermorgen
- in X Tagen (e.g., "in 37 Tagen")
- nächsten Montag, am Dienstag

## Temporary Controls

Works for: light, cover, switch, fan, automation

| Command | What Happens |
|---------|--------------|
| "Licht für 5 Minuten an" | On for 5 min, then restores |
| "Türklingel für 30 Minuten aus" | Off for 30 min, then restores |
| "Rollo auf 50% für 10 Minuten" | Sets position, then restores |

Uses `timebox_entity_state` script to snapshot and restore state.

## Multi-Command Splitting

The clarification capability splits compound commands:

| Input | Splits Into |
|-------|-------------|
| "Licht an und Rollo runter" | 2 commands |
| "Licht im Bad an und Küche aus" | 2 commands |
| "Rolläden im OG und UG runter" | 2 commands |

## Implicit Commands

Clarification transforms implicit requests:

| Input | Becomes |
|-------|---------|
| "Es ist zu dunkel" | "Mache das Licht heller" |
| "Zu hell hier" | "Mache das Licht dunkler" |
| "Im Büro ist es zu dunkel" | "Mache das Licht im Büro heller" |

---

# Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│ Stage 0: NLU Recognition        │
│ (Home Assistant built-in)       │
└─────────────────────────────────┘
    │ (if no match)
    ▼
┌─────────────────────────────────┐
│ Stage 1: Semantic Cache         │
│ (vector-validated lookup)     │
└─────────────────────────────────┘
    │ (if cache miss)
    ▼
┌─────────────────────────────────┐
│ Stage 2: Local LLM              │
│ (Ollama intent parsing)         │
└─────────────────────────────────┘
    │ (if unresolved or chat)
    ▼
┌─────────────────────────────────┐
│ Stage 3: Gemini Cloud           │
│ (Fallback + conversation)       │
└─────────────────────────────────┘
```

## Capabilities

| Capability | Purpose |
|------------|---------|
| `semantic_cache` | Fast cache lookup with vector validation |
| `clarification` | Split compound commands, transform implicit commands |
| `keyword_intent` | Detect domain and extract intent/slots via LLM |
| `entity_resolver` | Find entity IDs matching criteria |
| `intent_executor` | Execute HA intents, handle timebox |
| `calendar` | Multi-turn calendar event creation |
| `timer` | Timer management with device notifications |
| `vacuum` | Vacuum-specific slot extraction |
| `memory` | Store learned area/entity aliases |
| `area_resolver` | Fuzzy match area names |

## Floor vs Area

The system distinguishes floors from rooms:

**Floors** (`floor` slot): Erdgeschoss, EG, Obergeschoss, OG, Untergeschoss, UG, Keller

**Areas** (`area` slot): Küche, Bad, Büro, Schlafzimmer, etc.

---

# Troubleshooting

## Timebox "invalid slug" Error

**Cause:** HA ULID generates uppercase, but slugs must be lowercase.

**Fix:** Updated to pass lowercase UUID as `scene_id`.

## Entity Not Found

1. Check entity is exposed to conversation assistant
2. Verify area assignment
3. Try exact device name

## LLM Connection Failed

1. Verify Ollama is running
2. Check OLLAMA_HOST setting
3. Test: `curl http://HOST:11434/api/tags`

## Debug Logging

```yaml
logger:
  logs:
    custom_components.multistage_assist: debug
```
