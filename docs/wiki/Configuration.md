# Configuration Reference

Complete configuration options for MultiStage Assist.

## Basic Setup (Config Flow)

Configure via: **Settings → Devices & Services → Add Integration → MultiStage Assist**

| Option | Description | Default |
|--------|-------------|---------|
| **Ollama Host** | IP/hostname for Ollama server | `localhost` |
| **Ollama Port** | Ollama API port | `11434` |
| **Ollama Model** | LLM model for intent parsing | `qwen3:4b` |
| **Google API Key** | Gemini API key for Stage 3 | (required) |
| **Gemini Model** | Gemini model name | `gemini-2.0-flash` |
| **Reranker Host** | IP/hostname for reranker add-on | `localhost` |
| **Reranker Port** | Reranker API port | `5000` |

## Expert Settings (YAML)

Add to `configuration.yaml` for advanced tuning:

```yaml
multistage_assist:
  # --- Semantic Cache Tuning ---
  reranker_threshold: 0.73     # Minimum reranker score for cache hit (0.0-1.0)
  hybrid_enabled: true          # Enable hybrid keyword + vector search
  hybrid_alpha: 0.7             # Weight: 0.0 = keyword only, 1.0 = vector only
  hybrid_ngram_size: 2          # N-gram size for BM25 (1=words, 2=bigrams)
  vector_search_threshold: 0.5  # Minimum vector similarity for candidates
  vector_search_top_k: 10       # Max candidates from vector search
  
  # --- Cache Behavior ---
  cache_regenerate_on_startup: true  # Regenerate anchors on HA restart
  cache_max_entries: 10000           # Maximum user-learned cache entries
  
  # --- Low-Hardware Mode ---
  skip_stage1_llm: false  # Cache-only mode (see below)
  
  # --- LLM Behavior ---
  llm_timeout: 30         # Timeout in seconds for LLM calls
  llm_max_retries: 2      # Retry count on LLM failure
  
  # --- Debugging ---
  debug_cache_hits: false          # Log detailed cache hit/miss info
  debug_llm_prompts: false         # Log full LLM prompts and responses
  debug_intent_resolution: false   # Log intent resolution steps
```

## Cache-Only Mode

For low-hardware systems, enable `skip_stage1_llm: true`:

- Stage 1 uses **only** the semantic cache
- Commands not in cache escalate directly to Stage 3 (Gemini)
- No local LLM calls = faster processing
- Tradeoff: New command variations won't work until learned

## Required Scripts

Copy these scripts to your HA config for temporary/delayed controls:

| Script | Purpose | Location |
|--------|---------|----------|
| `timebox_entity_state` | Temporary controls ("für 5 Minuten") | `scripts/timebox_entity_state.yaml` |
| `delay_action` | Delayed controls ("in 10 Minuten") | `scripts/delay_action.yaml` |

After copying, reload scripts: **Developer Tools → YAML → Reload Scripts**

## Embedding Model

The semantic cache requires an embedding model. Pull it before first use:

```bash
ollama pull mxbai-embed-large
```

To use a different model:

```yaml
multistage_assist:
  embedding_model: "nomic-embed-text"  # Alternative model
```

## Debug Logging

```yaml
logger:
  logs:
    custom_components.multistage_assist: debug
```

### Key Loggers

| Component | What it logs |
|-----------|--------------|
| `stage0` | NLU recognition results |
| `stage1_cache` | Cache lookup hits/misses |
| `stage2_llm` | LLM intent parsing |
| `stage3_gemini` | Gemini API calls |
| `semantic_cache` | Cache operations |
| `entity_resolver` | Entity matching |
| `intent_executor` | Intent execution |
