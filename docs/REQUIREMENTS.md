# MultiStage Assist — Requirements

> **Source of truth** for all functional requirements.
> Every feature, bug fix, or change MUST trace back to a requirement here.
> If no requirement exists, add one before implementing.

---

## How to Use This Document

1. **Before implementing**: Find the relevant REQ-* IDs. If none exist, add new ones.
2. **In commit messages**: Reference REQ-* IDs (e.g. `fix: cover step logic (REQ-EXEC-009)`).
3. **In test docstrings**: Reference which REQ-* the test covers.
4. **When reviewing**: Verify the requirement is tested and documented.

---

## 1. Pipeline Architecture

### REQ-PIPE-001: 4-Stage Escalation
Sequential escalation through exactly 4 stages. Each returns `StageResult` with one of 6 statuses.
- Stage 0: HA built-in NLU (hassil) — no LLM
- Stage 1: Semantic cache (ChromaDB + BM25 hybrid)
- Stage 2: Local LLM via Ollama (default: `qwen3:4b-q4_K_M`)
- Stage 3: Cloud LLM — polymorphic (Gemini/OpenAI/Anthropic/Grok)
- **Source**: `conversation.py`, `stage_result.py`, `docs/ARCHITECTURE.md`

### REQ-PIPE-002: Minimalist Escalate Philosophy
Prefer the cheapest/fastest path. Each stage escalates only when unable to resolve with sufficient certainty.
- **Source**: `docs/ARCHITECTURE.md`

### REQ-PIPE-003: Stage 0 — HA Built-in NLU
Uses hassil `recognize_best()`. No LLM. Escalates on no match or >10 candidates.
- **Source**: `stage0.py`

### REQ-PIPE-004: Stage 0 Dry-run Recognition
Performs dry-run NLU recognition without executing intent. Returns normalized entities dict.
- **Source**: `stage0.py` `_dry_run_recognize()`, `_extract_params()`

### REQ-PIPE-005: Stage 1 — Semantic Cache Lookup
3-tier lookup: anchor check (fastest) → local fuzzy → remote add-on fallback.
- **Source**: `stage1_cache.py`, `capabilities/semantic_cache.py`

### REQ-PIPE-006: Stage 1 Preprocessing
Runs ImplicitIntentCapability + AtomicCommandCapability before cache lookup.
- **Source**: `stage1_cache.py`

### REQ-PIPE-007: Stage 1 Cache Bypass Intents
`CACHE_BYPASS_INTENTS`: HassTimerCancel. These always escalate to LLM.
- **Source**: `stage1_cache.py`

### REQ-PIPE-008: Stage 1 Multi-Command Return
If atomic command returns >1 command, returns `multi_command` status.
- **Source**: `stage1_cache.py`

### REQ-PIPE-009: Stage 2 — Local LLM Intent Parsing
Uses Ollama LLM for intent, domain, and slot extraction. Multi-turn for Timer/Calendar/Vacuum.
- **Source**: `stage2_llm.py`

### REQ-PIPE-010: Stage 2 Area Alias Learning
If area resolution returns `unknown_area`, returns `pending` status asking user for clarification.
- **Source**: `stage2_llm.py`

### REQ-PIPE-011: Stage 3 — Polymorphic Cloud LLM
4 providers: Gemini, OpenAI, Anthropic, Grok. Up to 5 reasoning turns. 9 MCP tools.
- **Source**: `stage3_cloud.py`, `capabilities/mcp.py`

### REQ-PIPE-012: Stage 3 Muscle Memory
Stage 3 can store successful resolutions back to Stage 1 via `store_cache_entry` MCP tool.
- **Source**: `stage3_cloud.py`, `capabilities/mcp.py`

### REQ-PIPE-013: Conversation Orchestration
Sequential: exit check → cleanup stale → ownership check → pending timeout → stages 0→1→2→3.
- **Source**: `conversation.py`

### REQ-PIPE-014: Global Exit Commands
"Abbruch", "Stop", etc. immediately clear pending state and return abort response.
- **Source**: `conversation.py`, `utils/german_utils.py` `EXIT_COMMANDS`

### REQ-PIPE-015: Pending Timeout Management
15s before re-ask, max 2 retries (30s total), absolute 5min cleanup (300s).
- **Source**: `conversation.py` `PENDING_TIMEOUT_SECONDS`, `PENDING_MAX_RETRIES`, `PENDING_ABSOLUTE_TIMEOUT`

### REQ-PIPE-016: StageResult Contract
6 statuses: `success`, `escalate`, `escalate_chat`, `multi_command`, `pending`, `error`.
- **Source**: `stage_result.py`

---

## 2. Semantic Cache

### REQ-CACHE-001: Dual-Mode Operation
Normal mode (external add-on + local) and add-on-only mode.
- **Source**: `capabilities/semantic_cache.py`

### REQ-CACHE-002: 3-Tier Lookup
Anchor check → local fuzzy → remote add-on fallback.
- **Source**: `capabilities/semantic_cache.py`, `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-003: Principle 1 — Grammatical Plurality
"Licht" ≠ "Lichter". Normalization MUST NOT collapse plural forms.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-004: Principle 2 — Spatial Distinction
"im Keller" MUST be distinct from "im Erdgeschoss" in vector space.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-005: Principle 3 — Intent Differentiation
"Schalte an" MUST NEVER match "Schalte aus". State is critical.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-006: Principle 4 — Domain Representation
Domains MUST remain distinct. Stripping domains collapses action-only entries.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-007: Principle 5 — Logical Opposites
Open MUST NOT be semantically identical to close.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-008: Principle 6 — Multi-Command Escalation
"und" or "," in input → `[MULTIPLE_COMMANDS_ESCALATION]` token → escalate to LLM.
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-009: Principle 7 — Number Normalization
Numbers normalized to centroids: "37%" and "82%" both → "50 Prozent".
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-010: Principle 8 — Multi-Word Names
Support "Kinder Badezimmer", "Kinderzimmer 1", "Wohn-Esszimmer".
- **Source**: `docs/CACHE_PRINCIPLES.md`

### REQ-CACHE-011: Variable Slot Stripping
Timer/calendar entries cached with variable slots stripped (duration, description, dates).
Intent pattern is cached; execution re-extracts variable data from original text.
- **Source**: `capabilities/semantic_cache.py` `_VARIABLE_SLOTS`

### REQ-CACHE-012: Cache Skip Filters
Skip caching for: short texts <3 words, disambiguation responses, relative brightness.
- **Source**: `capabilities/semantic_cache.py`, `capabilities/command_processor.py` `NOCACHE_INTENTS`

### REQ-CACHE-013: User Learning
Successful execution stores command to cache for future learning.
- **Source**: `execution_pipeline.py`, `capabilities/command_processor.py`

### REQ-CACHE-014: Expert Thresholds
`vector_search_threshold` (0.75), `hybrid_alpha` (0.7), `cache_max_entries` (10000).
- **Source**: `const.py` `EXPERT_DEFAULTS`

---

## 3. Capabilities

### REQ-CAP-001: Base Class Contract
All capabilities extend `Capability` from `capabilities/base.py`. Contract: `name`, `description`, `async run(user_input, **kwargs)`.
- **Source**: `capabilities/base.py`

### REQ-CAP-002: Automatic Dependency Injection
`BaseStage` auto-injects KnowledgeGraph, AreaResolver, Memory into capabilities.
- **Source**: `base_stage.py`

### REQ-CAP-003: EntityResolver
Resolves entities from domain, area, floor, name, device_class. Fuzzy matching with `_FUZZ_STRONG=92`, `_FUZZ_FALLBACK=84`. Filters by exposure and KG dependencies.
- **Source**: `capabilities/entity_resolver.py`

### REQ-CAP-004: AreaResolver
Fuzzy-matches area names. Integrates KnowledgeGraph alias learning.
- **Source**: `capabilities/area_resolver.py`

### REQ-CAP-005: KeywordIntent
LLM-based intent + domain + slot extraction. Keywords detect domain first, then LLM.
- **Source**: `capabilities/keyword_intent.py`

### REQ-CAP-006: IntentExecutor
`BRIGHTNESS_STEP=35%`, `COVER_STEP=25%`, min step 10%. From OFF → 30%. KG prerequisite resolution. Timebox/delay script support.
- **Source**: `capabilities/intent_executor.py`

### REQ-CAP-007: IntentConfirmation
German confirmation templates from `messages_de.py`. Randomized for variety.
- **Source**: `capabilities/intent_confirmation.py`

### REQ-CAP-008: CommandProcessor
Orchestrates: state filter → plural detect → disambiguate → execute → confirm → cache.
- **Source**: `capabilities/command_processor.py`

### REQ-CAP-009: ImplicitIntent
Transforms "zu dunkel" → "Licht heller". Fast path via `IMPLICIT_INTENT_MAPPINGS`, fallback to LLM.
- **Source**: `capabilities/implicit_intent.py`

### REQ-CAP-010: AtomicCommand
Splits on "und", "dann", commas. Multi-area detection.
- **Source**: `capabilities/atomic_command.py`

### REQ-CAP-011: MultiTurn Base
Mandatory/optional fields, auto-prompting, pending state management.
- **Source**: `capabilities/multi_turn_base.py`

### REQ-CAP-012: Timer
Duration mandatory. Optional: device_name, timer_name. Creates HA timer with device notification.
- **Source**: `capabilities/timer.py`

### REQ-CAP-013: Calendar
Summary + datetime + calendar mandatory. Relative dates. Confirmation preview.
- **Source**: `capabilities/calendar.py`

### REQ-CAP-014: Vacuum
Mode: vacuum (saugen) / mop (wischen). Scope: room / floor / global.
- **Source**: `capabilities/vacuum.py`

### REQ-CAP-015: KnowledgeGraph
Relationships: POWERED_BY, COUPLED_WITH, LUX_SENSOR, ASSOCIATED_COVER, ENERGY_PARENT. Learned aliases.
- **Source**: `capabilities/knowledge_graph.py`

### REQ-CAP-016: MCP Tools
9 tools: list_areas, list_entities, get_entity_details, list_automations, get_automation_details, store_personal_data, get_personal_data, get_system_capabilities, store_cache_entry.
- **Source**: `capabilities/mcp.py`

---

## 4. Execution

### REQ-EXEC-001: ExecutionPipeline Entry
Success status → `ExecutionPipeline.execute()`: filter → plural → disambiguate → execute → confirm → cache.
- **Source**: `execution_pipeline.py`

### REQ-EXEC-002: State-Aware Filtering
Turn off → only ON entities. Turn on → only OFF. GetState → no filter.
- **Source**: `execution_pipeline.py`

### REQ-EXEC-003: Global Query Handling
`HassGetState` with no entity_ids → all exposed entities in target domain.
- **Source**: `execution_pipeline.py`

### REQ-EXEC-004: Plural Detection
"alle", "beide" → execute on all candidates without disambiguation.
- **Source**: `capabilities/command_processor.py`

### REQ-EXEC-005: Disambiguation
2+ candidates → ask user. Pending state with candidates map. User response to `continue_disambiguation()`.
- **Source**: `capabilities/disambiguation.py`, `capabilities/command_processor.py`

### REQ-EXEC-006: KG Prerequisite Resolution
Check POWERED_BY/COUPLED_WITH before execution. AUTO mode: turn on dependency. WARN mode: inform user.
- **Source**: `capabilities/intent_executor.py`

### REQ-EXEC-007: Parameter Normalization
"halb" → 50, "viertel" → 25, percentages: "25%", "25 Prozent".
- **Source**: `capabilities/intent_executor.py`

### REQ-EXEC-008: Timebox Support
"für X Minuten" → `script.timebox_entity_state` with state snapshot + restore.
- **Source**: `capabilities/intent_executor.py`, `scripts/timebox_entity_state.yaml`

### REQ-EXEC-009: Delay Support
"in X Minuten" → `script.delay_action` with target action + delay.
- **Source**: `capabilities/intent_executor.py`, `scripts/delay_action.yaml`

### REQ-EXEC-010: Cache Storage on Success
Store to cache after successful execution. Skip for `NOCACHE_INTENTS`.
- **Source**: `capabilities/command_processor.py`

---

## 5. Internationalization

### REQ-I18N-001: German-First Centralization
ALL user-facing strings in `constants/messages_de.py`. NO hardcoded German in capabilities.
- **Source**: `.github/copilot-instructions.md`

### REQ-I18N-002: Umlaut Handling
Use `normalize_umlauts()` from `utils/german_utils.py`. Never inline `.replace()` chains.
- **Source**: `utils/german_utils.py`

### REQ-I18N-003: Data-Driven Constants
Domain keywords, area aliases, global scope words in `constants/` or `const.py`. Never inline.
- **Source**: `constants/area_keywords.py`, `constants/entity_keywords.py`, `constants/messages_de.py`

---

## 6. Configuration

### REQ-CONF-001: Service Domains
`SERVICE_DOMAINS = {"notify", "tts", "script", "automation", "scene"}` — bypass exposure filtering.
- **Source**: `const.py`

### REQ-CONF-002: Light-Compatible Domains
`LIGHT_COMPATIBLE_DOMAINS = {"light", "switch", "input_boolean"}`.
- **Source**: `const.py`

### REQ-CONF-003: Expert Defaults
13 tunable parameters in `EXPERT_DEFAULTS` dict. See `docs/wiki/Configuration.md`.
- **Source**: `const.py`

### REQ-CONF-004: Stage 3 Provider Selection
`CONF_STAGE3_PROVIDER`: "gemini" | "openai" | "anthropic" | "grok".
- **Source**: `const.py`, `config_flow.py`

---

## 7. Domain Support

### REQ-DOM-001: Light
Intents: TurnOn, TurnOff, LightSet, GetState, TemporaryControl. Brightness: 0-100, fractions.
- **Source**: `constants/domain_config.py`, `capabilities/intent_executor.py`

### REQ-DOM-002: Cover
Intents: TurnOn (open), TurnOff (close), SetPosition, GetState, TemporaryControl, DelayControl.
- **Source**: `constants/domain_config.py`

### REQ-DOM-003: Switch
Intents: TurnOn, TurnOff, GetState, TemporaryControl.
- **Source**: `constants/domain_config.py`

### REQ-DOM-004: Climate
Intents: SetTemperature, TurnOn, TurnOff, GetState.
- **Source**: `constants/domain_config.py`

### REQ-DOM-005: Fan
Intents: TurnOn, TurnOff, GetState, TemporaryControl.
- **Source**: `constants/domain_config.py`

### REQ-DOM-006: Sensor
Intents: GetState (read-only).
- **Source**: `constants/domain_config.py`

### REQ-DOM-007: Timer
Intents: TimerSet (duration mandatory), TimerCancel.
- **Source**: `capabilities/timer.py`

### REQ-DOM-008: Calendar
Intents: CalendarCreate (summary + datetime + calendar mandatory).
- **Source**: `capabilities/calendar.py`

### REQ-DOM-009: Vacuum
Intents: VacuumStart (room + mode mandatory).
- **Source**: `capabilities/vacuum.py`

### REQ-DOM-010: Automation
Intents: TurnOn, TurnOff, TemporaryControl.
- **Source**: `constants/domain_config.py`

---

## 8. Testing

### REQ-TEST-001: Test Runner
`bash scripts/run_tests.sh` — auto venv, deps, env vars. `--all` for Ollama tests.
- **Source**: `scripts/run_tests.sh`

### REQ-TEST-002: Every Change Needs a Test
Bug fixes and features MUST have a test covering the scenario.
- **Source**: `.github/copilot-instructions.md`

### REQ-TEST-003: Embedding Principles Verification
`test_embedding_principles.py` encodes all 8 cache principles as assertions.
- **Source**: `tests/test_embedding_principles.py`, `docs/CACHE_PRINCIPLES.md`

### REQ-TEST-004: Unit vs Integration Separation
Unit tests in `tests/`, integration (LLM-dependent) in `tests/integration/`.
- **Source**: `scripts/run_tests.sh`

---

## 9. Code Graph

### REQ-GRAPH-001: AST-Based Code Graph
SQLite-backed structural graph built from Python AST. Nodes: files, classes, functions, constants. Edges: imports, extends, calls, tests, stage_uses. Incremental rebuild via SHA-256 hash.
- **Source**: `graph/build_graph.py`

### REQ-GRAPH-002: Requirements as Structured Data
All REQ-* requirements stored in `graph/requirements.yaml` (machine-readable). Graph builder loads them into SQLite for cross-referencing with source and test files.
- **Source**: `graph/requirements.yaml`

### REQ-GRAPH-003: MCP Query Tools
MCP server exposes graph query tools: file purpose, blast radius, requirement lookup, capability info, stage overview, test mapping, sync values, search.
- **Source**: `graph/graph_server.py`

### REQ-GRAPH-004: Documentation Contract in Graph
The documentation contract (code change → docs to update) is encoded as graph edges so blast-radius queries automatically surface which docs need updating.
- **Source**: `graph/build_graph.py`

---

## 10. Code Quality

### REQ-QUAL-001: No Bare Excepts
All `except:` clauses must specify the exception type. Bare excepts hide `KeyboardInterrupt`, `SystemExit`, and real errors.
- **Source**: All `.py` files

### REQ-QUAL-002: Safe Tool Argument Parsing
When parsing LLM-provided tool arguments, malformed args must not silently default to empty dict and proceed with tool execution. Log the error and skip the tool call.
- **Source**: `stage3_cloud.py`, `capabilities/mcp.py`

### REQ-QUAL-003: No Dead Code
Unused capabilities, constants, and modules must be removed. Code that is defined but never imported or used in the pipeline is dead code.
- **Source**: All `.py` files

### REQ-QUAL-004: Consistent Import Style
All intra-package imports must use relative imports (`from ..` / `from .`). No absolute `from custom_components.multistage_assist` imports in production code.
- **Source**: All `capabilities/*.py` files

### REQ-QUAL-005: DRY — No Duplicated Constants
German strings, question templates, and keyword lists must have a single source of truth. No copy-paste between `constants/messages_de.py` and `utils/response_builder.py`.
- **Source**: `constants/messages_de.py`, `utils/response_builder.py`

### REQ-QUAL-006: Function Size Limit
Functions should not exceed ~150 lines. Large orchestration functions should extract domain-specific handlers into private methods.
- **Source**: `capabilities/intent_executor.py`

### REQ-QUAL-007: Shared Test Fixtures
Common test helpers (e.g., `make_input()`, `mock_async_handle` setup) must be defined once in `tests/conftest.py`, not duplicated across test files.
- **Source**: `tests/conftest.py`

### REQ-QUAL-008: MCP Loop Deduplication
The MCP multi-turn LLM tool-calling loop pattern must be a single parameterized method, not duplicated.
- **Source**: `capabilities/mcp.py`
