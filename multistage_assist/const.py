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