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

# Expert Settings (YAML only, not in config flow UI)
# These allow power users to fine-tune semantic cache behavior
CONF_RERANKER_THRESHOLD = "reranker_threshold"
CONF_HYBRID_ENABLED = "hybrid_enabled"
CONF_HYBRID_ALPHA = "hybrid_alpha"  # 0.0-1.0, weight for semantic vs keyword
CONF_HYBRID_NGRAM_SIZE = "hybrid_ngram_size"  # 1=words, 2=bigrams, 3=trigrams
CONF_VECTOR_THRESHOLD = "vector_search_threshold"
CONF_VECTOR_TOP_K = "vector_search_top_k"