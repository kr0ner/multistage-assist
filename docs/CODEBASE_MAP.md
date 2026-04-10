# MultiStage Assist — Codebase Map

> Quick-reference for LLM agents navigating this codebase.
> When you need to find where something is implemented, start here.

---

## Project Layout

```
multistage-assist/                  # HA custom component root
├── conversation.py                 # ORCHESTRATOR: 4-stage pipeline, pending state, history
├── execution_pipeline.py           # EXECUTION: KG filter → disambiguate → execute → confirm → cache
├── stage_result.py                 # CONTRACT: StageResult (6 statuses: success/escalate/escalate_chat/multi_command/pending/error)
├── base_stage.py                   # BASE: Stage orchestrator with auto-DI for capabilities
├── stage0.py                       # STAGE 0: HA built-in NLU (hassil), no LLM
├── stage1_cache.py                 # STAGE 1: Semantic cache lookup (ChromaDB + BM25)
├── stage2_llm.py                   # STAGE 2: Local LLM via Ollama
├── stage3_cloud.py                 # STAGE 3: Cloud LLM (Gemini/OpenAI/Anthropic/Grok)
├── const.py                        # CONSTANTS: Domains, expert defaults, config keys
├── config_flow.py                  # CONFIG UI: User-facing settings flow
├── prompt_executor.py              # LLM: Execution with escalation & schema validation
├── ollama_client.py                # LLM: REST client for Ollama
├── conversation_utils.py           # HELPERS: Response builders, duration parsing
├── capabilities/                   # ALL CAPABILITIES (see below)
├── constants/                      # ALL CONSTANTS (see below)
├── utils/                          # ALL UTILITIES (see below)
├── docs/                           # DOCUMENTATION
│   ├── REQUIREMENTS.md             # Source-of-truth requirements (REQ-* IDs)
│   ├── ARCHITECTURE.md             # Pipeline architecture & design
│   ├── CACHE_PRINCIPLES.md         # 8 embedding/normalization principles
│   ├── CODEBASE_MAP.md             # THIS FILE
│   └── wiki/                       # Feature docs (13 pages)
├── tests/                          # UNIT TESTS (no external deps)
│   ├── conftest.py                 # Shared fixtures (hass, entities, areas)
│   ├── scenario_fixtures.py        # Shared scenario test helpers
│   ├── scenarios_*.py              # End-to-end scenario definitions (7 files)
│   ├── integration/                # INTEGRATION TESTS (requires Ollama)
│   └── test_*.py                   # Unit test files (~44 files)
└── scripts/                        # HA scripts + test runner
    └── run_tests.sh                # TEST RUNNER (auto venv, deps, env)
```

---

## Capabilities Directory

| File | Class | Purpose | Key Method |
|------|-------|---------|------------|
| `base.py` | `Capability` | Base class | `run(user_input, **kwargs)` |
| `entity_resolver.py` | `EntityResolverCapability` | Entity discovery (area/fuzzy/alias) | `run(user_input, entities=slots)` |
| `area_resolver.py` | `AreaResolverCapability` | Area/floor matching | `run(user_input, area_name=...)` |
| `keyword_intent.py` | `KeywordIntentCapability` | Intent + domain extraction | `run(user_input)` → `{intent, domain, slots}` |
| `semantic_cache.py` | `SemanticCacheCapability` | Vector cache lookup + store | `lookup(text)`, `store(...)` |
| `knowledge_graph.py` | `KnowledgeGraphCapability` | Device relationships + aliases | `get_prerequisites(entity_id)` |
| `implicit_intent.py` | `ImplicitIntentCapability` | "too dark" → "brighter" | `run(user_input)` → `List[str]` |
| `atomic_command.py` | `AtomicCommandCapability` | Split "A und B" → [A, B] | `run(user_input)` → `List[str]` |
| `command_processor.py` | `CommandProcessorCapability` | Execute pipeline orchestration | `process(user_input, intent, ...)` |
| `intent_executor.py` | `IntentExecutorCapability` | Execute HA intents | `run(user_input, intent_name=...)` |
| `intent_confirmation.py` | `IntentConfirmationCapability` | German confirmation messages | `run(intent, entity_names)` |
| `intent_resolution.py` | `IntentResolutionCapability` | Keyword → Entity pipeline | `run(user_input)` |
| `disambiguation.py` | `DisambiguationCapability` | "Welches Licht?" questions | `run(candidates)` |
| `disambiguation_select.py` | `DisambiguationSelectCapability` | Parse user selection | `run(user_input, candidates)` |
| `plural_detection.py` | `PluralDetectionCapability` | Detect "alle"/"beide" | `run(user_input)` → `{multiple: bool}` |
| `multi_turn_base.py` | `MultiTurnCapability` | Multi-turn field collection | Base for Timer/Calendar |
| `timer.py` | `TimerCapability` | Timer creation | `run(user_input, intent_name=...)` |
| `calendar.py` | `CalendarCapability` | Calendar event creation | `run(user_input, intent_name=...)` |
| `vacuum.py` | `VacuumCapability` | Vacuum/mop control | `run(user_input, intent_name=...)` |
| `step_control.py` | `StepControlCapability` | Relative adjustments | `calculate_step(current, direction)` |
| `prompt_context.py` | `PromptContextBuilderCapability` | LLM context builder | `run(user_input, keywords=...)` |
| `mcp.py` | `McpToolCapability` | MCP tool registry (9 tools) | `get_tools()`, `execute(name, args)` |
| `llm_providers.py` | `LLMProvider` + concrete | Cloud LLM abstraction | `chat_completion(messages)` |

---

## Constants Directory

| File | Key Exports | Used By |
|------|-------------|---------|
| `messages_de.py` | `SYSTEM_MESSAGES`, `ERROR_MESSAGES`, `QUESTION_TEMPLATES`, `CONFIRMATION_TEMPLATES`, `GLOBAL_KEYWORDS`, `ORDINAL_MAP`, `COMMAND_STATE_MAP`, `ALL_KEYWORDS`, `NONE_KEYWORDS`, `EXIT_COMMANDS`, `IMPLICIT_INTENT_MAPPINGS` | All user-facing code |
| `entity_keywords.py` | `ENTITY_KEYWORDS` (10 domains), `ON_INDICATORS`, `OFF_INDICATORS`, `QUESTION_KEYWORDS`, `VACUUM_MOP_KEYWORDS`, `VACUUM_DRY_KEYWORDS`, `CALENDAR_GENERIC_TITLES`, `ENTITY_PLURALS` | keyword_intent, semantic_cache, vacuum |
| `area_keywords.py` | `AREA_ALIASES`, `FLOOR_KEYWORDS`, `AREA_PREPOSITIONS` | area_resolver, entity_resolver |
| `domain_config.py` | `DOMAIN_CONFIG` (German names, device words, intents, steps per domain) | keyword_intent, intent_executor |
| `sensor_units.py` | `DEVICE_CLASS_UNITS` (temperature→°C, power→W, etc.) | response_builder |

---

## Utils Directory

| File | Key Functions | Used By |
|------|---------------|---------|
| `german_utils.py` | `normalize_umlauts()`, `canonicalize()`, `EXIT_COMMANDS` | Everywhere (umlaut handling) |
| `fuzzy_utils.py` | `fuzzy_match_best()`, `levenshtein_distance()` | entity_resolver, keyword_intent |
| `duration_utils.py` | `parse_german_duration()` ("5 Minuten"→300s) | timer, calendar, intent_executor |
| `response_builder.py` | `join_names()`, `format_entity_list()` | intent_confirmation, execution |
| `json_utils.py` | `extract_json_from_llm_string()` | prompt_executor (LLM output parsing) |
| `semantic_cache_builder.py` | Anchor generation (4-tier: area/entity/floor/global) | semantic_cache init |
| `semantic_cache_types.py` | `CacheEntry` dataclass, defaults | semantic_cache |
| `service_discovery.py` | `get_entities_by_domain()` with exposure filter | entity_resolver, execution |
| `cache_patterns/` | Pattern library: `base.py`, `area.py`, `entity.py`, `global_patterns.py` | semantic_cache_builder |

---

## Data Flow

```
User utterance
    │
    ▼
conversation.py::async_process()
    │
    ├── Exit command check (EXIT_COMMANDS)
    ├── Cleanup stale pending states
    ├── Check ExecutionPipeline ownership (disambiguation/slot-fill)
    │   └── _handle_pending_execution() → continue_pending()
    ├── Check stage-level pending (multi-turn)
    │
    ▼ (no pending)
    _run_pipeline() → iterate stages:
    │
    ├── Stage 0: stage0.py::process()
    │   └── hassil recognize_best() → NLU match? → success/escalate
    │
    ├── Stage 1: stage1_cache.py::process()
    │   ├── ImplicitIntent: "zu dunkel" → "Licht heller"
    │   ├── AtomicCommand: "A und B" → [A, B] → multi_command
    │   └── SemanticCache lookup → hit? → success/escalate
    │
    ├── Stage 2: stage2_llm.py::process()
    │   ├── PromptContext: build entity/area context
    │   ├── KeywordIntent: LLM intent+domain+slots
    │   ├── AreaResolver: fuzzy area matching
    │   ├── Timer/Calendar/Vacuum: multi-turn
    │   └── EntityResolver: resolve to entity_ids
    │
    └── Stage 3: stage3_cloud.py::process()
        ├── Cloud LLM multi-turn (5 turns max)
        └── MCP tools for autonomous reasoning
    │
    ▼ (success)
execution_pipeline.py::execute()
    │
    ├── State-aware filtering (turn off → only ON entities)
    ├── Plural detection ("alle" → skip disambiguation)
    ├── Disambiguation ("Welches?" → pending)
    ├── KG prerequisite resolution (powered_by → auto turn on)
    ├── IntentExecutor: call HA service
    ├── IntentConfirmation: German response
    └── SemanticCache store: learn for future
```

---

## Key Constants (must stay in sync)

| Constant | Location | Current Value |
|----------|----------|---------------|
| `BRIGHTNESS_STEP` | `capabilities/intent_executor.py` | 35% |
| `COVER_STEP` | `capabilities/intent_executor.py` | 25% |
| `PENDING_TIMEOUT_SECONDS` | `conversation.py` | 15 |
| `PENDING_MAX_RETRIES` | `conversation.py` | 2 |
| `PENDING_ABSOLUTE_TIMEOUT` | `conversation.py` | 300 |
| `vector_search_threshold` | `const.py` EXPERT_DEFAULTS | 0.75 |
| `hybrid_alpha` | `const.py` EXPERT_DEFAULTS | 0.7 |
| `cache_max_entries` | `const.py` EXPERT_DEFAULTS | 10000 |
| `SERVICE_DOMAINS` | `const.py` | notify, tts, script, automation, scene |
| `LIGHT_COMPATIBLE_DOMAINS` | `const.py` | light, switch, input_boolean |
| `CACHE_BYPASS_INTENTS` | `stage1_cache.py` | HassTimerCancel |
| `Stage 3 max_turns` | `stage3_cloud.py` | 5 |

---

## Finding Things

| I want to... | Look at |
|--------------|---------|
| Add a new domain | `constants/domain_config.py` + `constants/entity_keywords.py` + `docs/wiki/Home.md` |
| Add a German string | `constants/messages_de.py` |
| Add an area alias | `constants/area_keywords.py` |
| Fix entity resolution | `capabilities/entity_resolver.py` |
| Fix area matching | `capabilities/area_resolver.py` |
| Change cache behavior | `capabilities/semantic_cache.py` + `docs/CACHE_PRINCIPLES.md` |
| Add a capability | `capabilities/base.py` (extend) + stage (register) + `docs/ARCHITECTURE.md` |
| Add an MCP tool | `capabilities/mcp.py` |
| Change LLM prompts | The capability that owns the prompt (see table above) |
| Fix confirmation text | `constants/messages_de.py` CONFIRMATION_TEMPLATES |
| Add a test | `tests/test_*.py` (unit) or `tests/integration/` (LLM) |
| Run tests | `bash scripts/run_tests.sh` |
| Change expert settings | `const.py` EXPERT_DEFAULTS + `docs/wiki/Configuration.md` |
| Debug pipeline flow | `conversation.py` + enable `debug_*` flags in expert settings |
