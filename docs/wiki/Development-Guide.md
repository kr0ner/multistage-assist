# Development Guide

Guide for developers contributing to MultiStage Assist.

## Project Structure

```
multistage_assist/
├── __init__.py            # Integration setup
├── conversation.py        # Main orchestrator (4-stage pipeline)
├── conversation_utils.py  # Response helpers, duration parsing
├── execution_pipeline.py  # Unified execution for all stages
├── stage0.py              # Stage 0: NLU (Home Assistant built-in)
├── stage1_cache.py        # Stage 1: Semantic cache lookup
├── stage2_llm.py          # Stage 2: Local LLM (Ollama)
├── stage3_cloud.py        # Stage 3: Cloud LLM (Gemini/OpenAI/Anthropic/Grok)
├── base_stage.py          # Base class for stages with DI
├── stage_result.py        # StageResult data class
├── const.py               # All configuration constants
├── config_flow.py         # HA config UI
├── ollama_client.py       # Local Ollama API client
├── prompt_executor.py     # Shared LLM prompt execution
├── capabilities/          # Modular capability implementations
│   ├── base.py                # Abstract Capability base class
│   ├── semantic_cache.py      # Cache lookup (vector + BM25 hybrid)
│   ├── keyword_intent.py      # LLM-based intent detection
│   ├── entity_resolver.py     # Entity ID resolution
│   ├── area_resolver.py       # Fuzzy area matching
│   ├── intent_executor.py     # HA intent execution + state verify
│   ├── intent_confirmation.py # Natural confirmation messages
│   ├── command_processor.py   # Execution orchestration
│   ├── knowledge_graph.py     # Device deps, aliases, personal data
│   ├── implicit_intent.py     # Vague command rephrasing
│   ├── atomic_command.py      # Compound command splitting
│   ├── mcp.py                 # MCP tool registry for LLM access
│   ├── prompt_context.py      # LLM prompt context builder
│   ├── timer.py               # Timer creation (multi-turn)
│   ├── calendar.py            # Calendar events (multi-turn)
│   ├── vacuum.py              # Vacuum room/mode parsing
│   ├── step_control.py        # Relative adjustments
│   ├── plural_detection.py    # Singular/plural detection
│   ├── disambiguation.py      # Entity selection
│   └── disambiguation_select.py
├── constants/             # Configuration constants
│   ├── domain_config.py       # Domain-specific config
│   ├── entity_keywords.py     # German domain keywords
│   ├── messages_de.py         # German response messages
│   ├── area_keywords.py       # Area aliases, prepositions, floors
│   └── sensor_units.py        # Sensor unit mappings
├── utils/                 # Utility modules
│   ├── german_utils.py        # Cache normalization, duration extraction
│   ├── fuzzy_utils.py         # Levenshtein matching
│   ├── response_builder.py    # German text generation
│   ├── duration_utils.py      # Duration parsing
│   ├── json_utils.py          # LLM response JSON extraction
│   ├── service_discovery.py   # HA service discovery
│   ├── semantic_cache_types.py # CacheEntry dataclass, constants
│   └── semantic_cache_builder.py # Anchor pattern generation
├── scripts/               # HA scripts for temp/delay actions
└── tests/                 # Test suite
```

## Pipeline Flow

```
User Input → conversation.py
    ↓
┌─────────────────────────────────────────────┐
│ Stage 0: NLU (stage0.py)                    │
│ • Home Assistant built-in intent recognition│
│ • Fast pattern matching                     │
│ → Returns success if NLU matches            │
└─────────────────────────────────────────────┘
    ↓ (escalate if no match)
┌─────────────────────────────────────────────┐
│ Stage 1: Cache (stage1_cache.py)            │
│ • Semantic cache lookup via cache add-on │
│ • Returns cached intent + entities          │
│ → Returns success if cache hit              │
└─────────────────────────────────────────────┘
    ↓ (escalate if cache miss)
┌─────────────────────────────────────────────┐
│ Stage 2: LLM (stage2_llm.py)                │
│ • Local Ollama LLM for intent parsing       │
│ • KeywordIntentCapability extracts slots    │
│ • EntityResolver finds entity IDs           │
│ → Returns success if intent resolved        │
└─────────────────────────────────────────────┘
    ↓ (escalate if unresolved or chat request)
┌─────────────────────────────────────────────┐
│ Stage 3: Cloud LLM (stage3_cloud.py)        │
│ • Polymorphic (Gemini/OpenAI/Anthropic/Grok)│
│ • Multi-turn MCP tool reasoning (5 turns)   │
│ • Chat mode for general conversation        │
└─────────────────────────────────────────────┘
    ↓
ExecutionPipeline → CommandProcessor → IntentExecutor
```

## StageResult Interface

Every stage returns a `StageResult`:

```python
from stage_result import StageResult

# Success: Ready for execution
return StageResult.success(
    intent="HassTurnOn",
    entity_ids=["light.kitchen"],
    params={"brightness": 50},
    context={"from_cache": True},
    raw_text=user_input.text,
)

# Escalate: Pass to next stage
return StageResult.escalate(
    context={"cache_miss": True},
    raw_text=user_input.text,
)

# Error: Return message to user
return StageResult.error(
    message="Entity not found",
    response=await make_response("Gerät nicht gefunden.", user_input),
)
```

## Adding a New Capability

1. **Create capability file** in `capabilities/`:

```python
from .base import Capability

class MyCapability(Capability):
    name = "my_capability"
    description = "What it does"
    
    async def run(self, user_input, **kwargs):
        # Implementation
        return result
```

2. **Register in stage** (e.g., `stage2_llm.py`):

```python
capabilities = [
    KeywordIntentCapability,
    MyCapability,  # Add here
]
```

3. **Use in stage processing**:

```python
result = await self.use("my_capability", user_input)
```

## Running Tests

Use the workflow: `/run-tests`

```bash
# Quick test run
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term

# Specific test file
pytest tests/test_semantic_cache.py -v
```

### Test Requirements

Tests require:
- Running Ollama instance (for LLM tests)
- cache add-on (for cache tests)
- Configure in `pytest.ini` or environment variables

## Key Conventions

- **German language**: All user-facing text in German
- **Logging**: Use `_LOGGER = logging.getLogger(__name__)`
- **Type hints**: All public methods should have type annotations
- **Docstrings**: Google-style docstrings for public APIs
