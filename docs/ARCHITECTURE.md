# Multi-Stage Assist: Architecture & Principles

## 1. Multi-Stage Pipeline Philosophy
The system follows a "minimalist escalate" pattern. We prefer the cheapest/fastest path that solves the user's intent with high confidence.

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 0: NLU (stage0.py)            │
│ • Home Assistant built-in (hassil)  │
│ • Fast pattern matching, no LLM     │
└─────────────────────────────────────┘
    │ (if no match or ambiguous)
    ▼
┌─────────────────────────────────────┐
│ Stage 1: Semantic Cache             │
│   (stage1_cache.py)                 │
│ • ChromaDB vectors + BM25 hybrid    │
│ • Pre-generated anchor patterns     │
│ • User-learned command cache        │
└─────────────────────────────────────┘
    │ (if cache miss)
    ▼
┌─────────────────────────────────────┐
│ Stage 2: Local LLM (stage2_llm.py)  │
│ • Ollama-based intent parsing       │
│ • Keyword Intent Detection          │
│ • Entity Resolution + MCP Recovery  │
└─────────────────────────────────────┘
    │ (if unresolved or chat)
    ▼
┌─────────────────────────────────────┐
│ Stage 3: Cloud LLM                  │
│   (stage3_cloud.py)                 │
│ • Polymorphic cloud providers       │
│ • Multi-turn MCP tool reasoning     │
│ • Free-form conversation            │
└─────────────────────────────────────┘
    │
    ▼
ExecutionPipeline → CommandProcessor → IntentExecutor
```

### Stage 0: NLU (stage0.py)
- **Goal**: Instant intent recognition using Home Assistant's built-in NLU (hassil).
- **No LLM calls** — purely regex/pattern-based.
- **Capabilities**: `EntityResolverCapability`, `AreaResolverCapability`, `KnowledgeGraphCapability`.
- **Escalates when**: No intent match found or ambiguous entity references.

### Stage 1: Fast-Path — Semantic Cache (stage1_cache.py)
- **Goal**: Immediate resolution for common/repeated commands.
- **Anchors**: Minimalist set of ~2 templates per intent/domain pair (e.g., 2 for `HassTurnOn light`, 2 for `HassTurnOn switch`).
- **Normalisation**: Inputs are article-stripped and collapsed into "centroids" (e.g., `helligkeit_einstellen`, `position_einstellen`) to maximize clustering density.
- **Embedding Model**: strictly `models/fine_tuned_smarthome_model`.
- **Lookup Hierarchy**: Anchor check (fastest) → Local fuzzy (learned entries) → Remote add-on fallback.
- **Capabilities**: `ImplicitIntentCapability`, `AtomicCommandCapability`, `SemanticCacheCapability`, `KnowledgeGraphCapability`, `EntityResolverCapability`.
- **Cache Bypass Intents**: `HassTimerSet`, `HassTimerCancel` (context-dependent, never cached).
- **Escalates when**: No cache hit above threshold (default: 0.75) or command requires fresh processing.

### Stage 2: Mid-Path — Local LLM (stage2_llm.py)
- **Goal**: Intent parsing for commands not in cache using local compute.
- **Model**: configurable (default `qwen3:4b-q4_K_M`).
- **Logic**: Keyword intent extraction → entity resolution → area learning detection.
- **MCP Recovery**: LLM-guided entity lookup for hard-to-resolve entities.
- **Capabilities**: `KeywordIntentCapability`, `EntityResolverCapability`, `AreaResolverCapability`, `KnowledgeGraphCapability`, `MultiTurnCapability`, `TimerCapability`, `CalendarCapability`, `McpToolCapability`, `PromptContextBuilderCapability`.
- **Escalates when**: Intent unresolved or chat mode detected.

### Stage 3: Cloud-Path — Reasoning & Chat (stage3_cloud.py)
- **Goal**: Complex queries, multi-device management, and general chat.
- **Providers**: Polymorphic — Gemini (default: `gemini-2.0-flash-lite`), OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-sonnet-latest`), Grok (`grok-2-1212`).
- **Tool Access**: Full Model Context Protocol (MCP) with 9 tools: `list_areas`, `list_entities`, `get_entity_details`, `list_automations`, `get_automation_details`, `store_personal_data`, `get_personal_data`, `get_system_capabilities`, `store_cache_entry`.
- **Muscle Memory**: Stage 3 can autonomously store successful resolutions back to Stage 1 via `store_cache_entry`. This allows the system to "learn" user phrasing patterns over time.
- **Multi-turn**: Up to 5 reasoning turns per request. Per-conversation message history (max 10 messages = 5 turns).
- **Capabilities**: `McpToolCapability`.

## 2. Conversation Orchestration (conversation.py)

The `MultiStageAssistAgent` orchestrates all stages sequentially:

1. **Exit commands**: Global abort ("Abbruch", "Stop") → clears pending state.
2. **Cleanup**: Remove stale pending states from other conversations.
3. **Ownership check**: If `ExecutionPipeline` owns the conversation (disambiguation, slot-filling) → route there.
4. **Pending timeout**: 15 seconds → re-ask question once, max 2 retries (30s total), absolute timeout 5 minutes.
5. **Run stages**: Stage 0 → 1 → 2 → 3, each escalating until success.
6. **Execute or chat**: On success → `ExecutionPipeline`, on chat → direct cloud response.

## 3. Execution Pipeline (execution_pipeline.py)

Once a stage resolves intent + entities with `status="success"`, the `ExecutionPipeline` takes over via `CommandProcessorCapability`:

1. **State-aware filtering** — Turn off → filter to only ON entities.
2. **Plural detection** — "alle" means no disambiguation.
3. **Disambiguation** — Ask user which device if multiple matches (up to 2 turns).
4. **Knowledge Graph preconditions** — Ensure dependent devices are activated (powered_by, coupled_with).
5. **Intent execution** — Call `IntentExecutorCapability` with state verification.
6. **Confirmation generation** — Build natural language response.
7. **Semantic cache storage** — Store successful command for future learning.

## 4. Stage Result Interface (stage_result.py)

All stages return a unified `StageResult` with one of 6 statuses:
- **success**: Intent + entities resolved → execute via `ExecutionPipeline`.
- **escalate**: Pass to next stage with enriched context (NLU hints, slots).
- **escalate_chat**: User wants to chat → Stage 3 chat-only mode.
- **multi_command**: Split commands detected → iterate each through pipeline.
- **pending**: Asking user for clarification (area, entity, etc.).
- **error**: Unrecoverable error with pre-built response.

## 5. Shared Capabilities (capabilities/)

Capabilities are reusable modules shared across stages. Each stage declares its capabilities list; `BaseStage` auto-injects dependencies (Memory, KnowledgeGraph, AreaResolver) into all capabilities via setter methods.

### Core Capabilities
| Capability | Purpose |
|---|---|
| `SemanticCacheCapability` | Vector-based semantic lookup with hybrid BM25 |
| `EntityResolverCapability` | Resolve entities by area, name, device class |
| `AreaResolverCapability` | Fuzzy area name matching (German) |
| `KeywordIntentCapability` | LLM-based intent/domain/slot extraction |
| `KnowledgeGraphCapability` | Persistent memory: device deps, user facts, area/entity aliases |
| `ImplicitIntentCapability` | Rephrase vague commands ("zu dunkel" → "Licht heller") |
| `AtomicCommandCapability` | Split compound commands on "und", "dann", commas |
| `CommandProcessorCapability` | Orchestrate execution: filter, disambiguate, execute, confirm, cache |
| `IntentExecutorCapability` | Execute HA intents with state verification |
| `IntentConfirmationCapability` | Generate natural confirmation messages |
| `McpToolCapability` | MCP tool registry for Stage 2/3 LLM tool access |
| `PromptContextBuilderCapability` | Build entity/area context for LLM prompts |

### Domain Capabilities
| Capability | Purpose |
|---|---|
| `TimerCapability` | Multi-turn timer creation (extends `MultiTurnCapability`) |
| `CalendarCapability` | Multi-turn calendar event creation (extends `MultiTurnCapability`) |
| `VacuumCapability` | LLM-based room/mode parsing for vacuum control |
| `StepControlCapability` | Relative adjustments (brightness ±35%, cover ±25%) |
| `PluralDetectionCapability` | Detect plural/singular references in German |

## 6. Localization & Strings
- **messages_de.py**: Single source of truth for all German linguistic representations.
- **NO hardcoded German strings** are allowed in the execution or capability layer.

## 7. Hardware Independence
- **Embedding Model**: Fixed to `multilingual-minilm` (same as the Add-on) for consistent vector spaces.
- **Inference**: Configurable via Ollama (Local) or API (Cloud).
- **Cache-Only Mode**: Set `skip_stage1_llm: true` to skip Stage 2 local LLM; cache misses escalate directly to Stage 3.

## 8. Development Principles
- **Avoid Over-Engineering**: Features should emerge from LLM reasoning rather than complex Python conditionals.
- **Test-Driven Requirements**: Every bug fix or feature must have an integration test covering the scenario.
- **Minimalist Anchors**: Do NOT expand the anchor database to solve misses. Instead, refine the normalization pipeline ("Train the model").
- **Keep Docs In Sync**: When modifying pipeline stages, capabilities, constants, or execution flow, update `docs/ARCHITECTURE.md`, `docs/CACHE_PRINCIPLES.md`, and the relevant `docs/wiki/` pages in the same commit. See `.github/copilot-instructions.md` for the full documentation contract.

## 9. Semantic Cache Principles
To maintain a high-quality vector space for German smart home commands, the normalization pipeline MUST adhere to the [Semantic Cache Core Principles](CACHE_PRINCIPLES.md). These principles are non-negotiable for anyone modifying Stage 1 normalization or the embedding processing logic.
