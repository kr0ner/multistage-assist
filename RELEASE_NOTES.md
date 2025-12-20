# Multi-Stage Assist - Release Notes

## v1.1.0 (2025-12-20)

### New Features

- **Generic Step Control Capability** (`capabilities/step_control.py`)
  - Domain-agnostic step_up/step_down for relative adjustments
  - Supports: light (brightness), cover (position), fan (percentage), climate (temperature)
  - 20 test cases in `tests/test_step_control.py`

- **Centralized Domain Configuration** (`constants/domain_config.py`)
  - Unified `DOMAIN_CONFIG` structure for all 11 domains
  - German names, device words, keywords, intents, step configs
  - Prepares for future multi-language support
  - Helper functions: `get_domain_name()`, `get_step_config()`, etc.

- **Centralized German Messages** (`constants/messages_de.py`)
  - 56 centralized German strings (errors, questions, confirmations)
  - Helper functions: `get_error_message()`, `get_confirmation()`, etc.

- **Expert Configuration Options** (`const.py`)
  - `skip_stage1_llm`: Cache-only mode for low-hardware systems
  - `cache_regenerate_on_startup`: Control anchor regeneration
  - `cache_max_entries`: Limit cache size
  - `llm_timeout`, `llm_max_retries`: LLM behavior tuning
  - `debug_cache_hits`, `debug_llm_prompts`, `debug_intent_resolution`: Debug flags
  - Added `EXPERT_DEFAULTS` dict with default values

- **Cache-Only Mode** (`stage1.py`)
  - When `skip_stage1_llm: true`, Stage1 uses only semantic cache
  - Cache misses escalate directly to Stage2
  - Ideal for low-hardware deployments

### Bug Fixes

- **Fixed false positive confirmation on verification failure**
  - Previously: System said "Küchenradio ist eingeschaltet." even when device failed to turn on
  - Now: Shows "{device} reagiert nicht." when verification times out
  - Also skips caching failed commands
  - Files: `intent_executor.py`, `command_processor.py`, `stage1.py`

- **Fixed hybrid search shape mismatch** (`semantic_cache.py`)
  - BM25 index could become stale (size mismatch with cache)
  - Added safety check and auto-rebuild of BM25 index
  - Falls back to vector-only search if rebuild fails

### Code Cleanup

- **Dead code removal**
  - Removed unused `State` import from `entity_resolver.py`, `knowledge_graph.py`
  - Removed unused `normalize_speech_for_tts` import from `intent_executor.py`
  - Removed unused `Callable` import from `service_discovery.py`
  - Removed unused `pre_state` parameter from `_verify_execution()`
  - Removed unused `previous_results` parameter from `_execute_sequence()`

### Documentation

- **README.md refactored** (192 → ~80 lines)
  - Now a brief pointer to wiki documentation
  - Added expert config examples
  - Wiki is single source of truth for detailed docs

- **TODO.md updated**
  - Added performance optimization roadmap
  - Documented bottleneck analysis (17.5s for compound command)
  - Added proposed Docker-based inference service architecture

---

## v1.0.0 (Initial Release)

- Multi-Stage Pipeline (Stage 0/1/2)
- Semantic Command Cache with Ollama embeddings
- Reranker integration (API and local modes)
- Adaptive Learning (Memory)
- Interactive Disambiguation
- German language optimization
