"""Constants for the Multi-Stage Assist integration."""

DOMAIN = "multistage_assist"

# Stage 1: Local Ollama/LLM for Intent Detection
CONF_STAGE1_IP = "stage1_ip"
CONF_STAGE1_PORT = "stage1_port"
CONF_STAGE1_MODEL = "stage1_model"

# Stage 2: Google Gemini for Chat
CONF_GOOGLE_API_KEY = "google_api_key"
CONF_STAGE2_MODEL = "stage2_model"  # e.g. "gemini-2.0-flash"

# Embedding: Ollama for Semantic Cache (defaults to stage1 settings)
CONF_EMBEDDING_IP = "embedding_ip"
CONF_EMBEDDING_PORT = "embedding_port"
CONF_EMBEDDING_MODEL = "embedding_model"

# Reranker: For semantic cache validation
CONF_RERANKER_IP = "reranker_ip"
CONF_RERANKER_PORT = "reranker_port"

# --- Expert Settings (YAML only, not in config flow UI) ---
# These allow power users to fine-tune behavior

# Semantic Cache Settings
CONF_RERANKER_THRESHOLD = "reranker_threshold"
CONF_HYBRID_ENABLED = "hybrid_enabled"
CONF_HYBRID_ALPHA = "hybrid_alpha"  # 0.0-1.0, weight for semantic vs keyword
CONF_HYBRID_NGRAM_SIZE = "hybrid_ngram_size"  # 1=words, 2=bigrams, 3=trigrams
CONF_VECTOR_THRESHOLD = "vector_search_threshold"
CONF_VECTOR_TOP_K = "vector_search_top_k"

# Cache Behavior Settings (NEW)
CONF_CACHE_REGENERATE_ON_STARTUP = "cache_regenerate_on_startup"  # Default: True
CONF_CACHE_MAX_ENTRIES = "cache_max_entries"  # Default: 10000

# Low Hardware Mode (NEW)
# When enabled, Stage1 relies ONLY on the semantic cache.
# Commands not learned (cached) will not be supported.
# This skips all LLM calls in Stage1 for faster processing on limited hardware.
CONF_SKIP_STAGE1_LLM = "skip_stage1_llm"  # Default: False

# LLM Behavior Settings (NEW)
CONF_LLM_TIMEOUT = "llm_timeout"  # Default: 30 seconds
CONF_LLM_MAX_RETRIES = "llm_max_retries"  # Default: 2

# Debugging Settings (NEW)
CONF_DEBUG_CACHE_HITS = "debug_cache_hits"  # Log cache hits/misses in detail
CONF_DEBUG_LLM_PROMPTS = "debug_llm_prompts"  # Log LLM prompts and responses
CONF_DEBUG_INTENT_RESOLUTION = "debug_intent_resolution"  # Log intent resolution steps

# Default values for expert settings
EXPERT_DEFAULTS = {
    CONF_RERANKER_THRESHOLD: 0.73,
    CONF_HYBRID_ENABLED: True,
    CONF_HYBRID_ALPHA: 0.7,
    CONF_HYBRID_NGRAM_SIZE: 2,
    CONF_VECTOR_THRESHOLD: 0.5,
    CONF_VECTOR_TOP_K: 10,
    CONF_CACHE_REGENERATE_ON_STARTUP: True,
    CONF_CACHE_MAX_ENTRIES: 10000,
    CONF_SKIP_STAGE1_LLM: False,
    CONF_LLM_TIMEOUT: 30,
    CONF_LLM_MAX_RETRIES: 2,
    CONF_DEBUG_CACHE_HITS: False,
    CONF_DEBUG_LLM_PROMPTS: False,
    CONF_DEBUG_INTENT_RESOLUTION: False,
}