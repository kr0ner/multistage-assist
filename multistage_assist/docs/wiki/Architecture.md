# Architecture

## Pipeline

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│ Stage 0: NLU Recognition        │
│ • Home Assistant built-in       │
│ • Fast pattern matching         │
└─────────────────────────────────┘
    │ (if no match)
    ▼
┌─────────────────────────────────┐
│ Stage 1: LLM Enhancement        │
│ • Clarification                 │
│ • Keyword Intent Detection      │
│ • Entity Resolution             │
│ • Intent Execution              │
└─────────────────────────────────┘
    │ (if unresolved)
    ▼
┌─────────────────────────────────┐
│ Stage 2: Chat Mode              │
│ • Free-form conversation        │
└─────────────────────────────────┘
```

## Stage 0: NLU

Uses Home Assistant's built-in `conversation.async_recognize`.

**Escalates when:**
- No intent match found
- Ambiguous entity references

## Stage 1: LLM Enhancement

### Processing Flow

1. **Clarification** - Split compound commands, transform implicit requests
2. **Keyword Intent** - Detect domain from keywords, extract intent/slots via LLM
3. **Intent Resolution** - Resolve areas, find entity IDs
4. **Entity Fallback** - If no domain found, fuzzy match entity names
5. **Intent Execution** - Execute HA intent, verify success
6. **Confirmation** - Generate natural language response

### Domain Detection

Domain is detected from keywords in the input:
- "licht", "lampe" → light
- "rollo", "jalousie" → cover
- "heizung", "thermostat" → climate
- "timer", "wecker" → timer
- etc.

### Entity Fallback

When keyword_intent finds no domain:
1. Extract potential entity name from input
2. Remove command words (an, aus, für, etc.)
3. Fuzzy match against all entity names
4. Execute if match score >= 80%

## Stage 2: Chat Mode

Free-form LLM conversation for unrecognized commands.

## Capabilities

| Capability | Purpose |
|------------|---------|
| `clarification` | Split/transform commands |
| `keyword_intent` | Detect domain/intent |
| `intent_resolution` | Resolve entities |
| `entity_resolver` | Find entity IDs |
| `intent_executor` | Execute + verify |
| `intent_confirmation` | Generate response |
| `calendar` | Calendar events |
| `timer` | Timer management |
| `vacuum` | Vacuum slots |
| `memory` | Learned aliases |
| `area_alias` | Area matching |

## LLM Configuration

```yaml
llm:
  model: "qwen3:4b-instruct"
  host: "192.168.178.108"
  port: 11434
```

## Entity Exposure

Only entities exposed to conversation assistant are considered.
Configure in: Settings → Voice Assistants → Expose

## Memory

Learned aliases stored in:
`/config/.storage/multistage_assist_memory.json`
