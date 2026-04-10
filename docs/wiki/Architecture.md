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
│ Stage 1: Semantic Cache         │
│ • vector-validated lookup     │
│ • Pre-generated anchor patterns │
│ • User-learned command cache    │
└─────────────────────────────────┘
    │ (if cache miss)
    ▼
┌─────────────────────────────────┐
│ Stage 2: Local LLM              │
│ • Ollama-based intent parsing   │
│ • Keyword Intent Detection      │
│ • Entity Resolution             │
└─────────────────────────────────┘
    │ (if unresolved or chat)
    ▼
┌─────────────────────────────────┐
│ Stage 3: Cloud LLM              │
│ • Polymorphic providers         │
│   (Gemini/OpenAI/Anthropic/Grok)│
│ • Multi-turn MCP tool reasoning │
│ • Free-form conversation        │
└─────────────────────────────────┘
```

## Stage 0: NLU

Uses Home Assistant's built-in `conversation.async_recognize`.

**Escalates when:**
- No intent match found
- Ambiguous entity references

## Stage 1: Semantic Cache

Fast path using pre-computed and learned command patterns.

**Features:**
- **Anchor patterns** - Pre-generated from entity/area combinations
- **User learning** - Successful commands are cached for replay
- **vector validation** - External add-on validates cache matches
- **Hybrid search** - Combines vector similarity + BM25 keyword matching

**Escalates when:**
- No cache hit above threshold (default: 0.73)
- Command requires fresh processing (timers, delayed controls)

## Stage 2: Local LLM

Local Ollama-based intent parsing for commands not in cache.

### Processing Flow

1. **Clarification** - Split compound commands, transform implicit requests
2. **Keyword Intent** - Detect domain from keywords, extract intent/slots via LLM
3. **Intent Resolution** - Resolve areas, find entity IDs
4. **Entity Fallback** - If no domain found, fuzzy match entity names

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

## Stage 3: Cloud LLM (stage3_cloud.py)

Polymorphic cloud LLM for edge cases and general conversation.

**Providers:** Gemini (default: `gemini-2.0-flash-lite`), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-sonnet-latest`), Grok (`grok-2-1212`).

**Handles:**
- Commands that local LLM couldn't resolve
- General chat/conversation (jokes, help, etc.)
- Complex multi-step reasoning with MCP tools (up to 5 turns)

**MCP Tools:** `list_areas`, `list_entities`, `get_entity_details`, `list_automations`, `get_automation_details`, `store_personal_data`, `get_personal_data`, `get_system_capabilities`, `store_cache_entry`.

## Capabilities

| Capability | Purpose |
|------------|---------|
| `implicit_intent` | Rephrase vague commands ("zu dunkel" → "Licht heller") |
| `atomic_command` | Split compound commands on "und", commas |
| `keyword_intent` | LLM-based intent/domain/slot extraction |
| `entity_resolver` | Resolve entities by area, name, device class |
| `area_resolver` | Fuzzy area name matching |
| `intent_executor` | Execute HA intents + state verification |
| `intent_confirmation` | Generate natural confirmation messages |
| `command_processor` | Orchestrate: filter, disambiguate, execute, confirm, cache |
| `semantic_cache` | Vector-based lookup with hybrid BM25 |
| `knowledge_graph` | Device deps, user facts, area/entity aliases |
| `mcp` | MCP tool registry for LLM tool access |
| `prompt_context` | Build entity/area context for LLM prompts |
| `calendar` | Multi-turn calendar event creation |
| `timer` | Multi-turn timer creation |
| `vacuum` | LLM-based room/mode parsing |
| `step_control` | Relative adjustments (brightness ±35%, cover ±25%) |
| `plural_detection` | Detect plural/singular references |
| `disambiguation` | Multi-turn entity selection |

## LLM Configuration

Configured via integration UI (Settings → Devices & Services → MultiStage Assist):
- **Ollama Host**: IP/hostname (default: `127.0.0.1`)
- **Ollama Port**: API port (default: `11434`)
- **Ollama Model**: default `qwen3:4b-q4_K_M`
- **Cloud Provider**: gemini, openai, anthropic, or grok
- **Cloud Model**: provider-specific default

See [Configuration](Configuration.md) for expert YAML settings.

## Entity Exposure

Only entities exposed to conversation assistant are considered.
Configure in: Settings → Voice Assistants → Expose

## Knowledge Graph & Memory

Learned aliases, device dependencies, and personal data stored via `KnowledgeGraphCapability` in:
`/config/.storage/multistage_assist_knowledge_graph.json`
