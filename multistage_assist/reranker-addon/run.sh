#!/usr/bin/env bash
set -e

# Read configuration from HA addon options
CONFIG_PATH=/data/options.json
MODEL=$(jq -r '.model' $CONFIG_PATH)
DEVICE=$(jq -r '.device' $CONFIG_PATH)
PORT=$(jq -r '.port' $CONFIG_PATH)
HF_HOME_PATH=$(jq -r '.HF_HOME' $CONFIG_PATH)

echo "[INFO] Starting Semantic Reranker addon..."
echo "[INFO] Model: $MODEL"
echo "[INFO] Device: $DEVICE"
echo "[INFO] Port: $PORT"
echo "[INFO] Model cache: $HF_HOME_PATH"

# Create model cache directory (like Ollama creates OLLAMA_MODELS dir)
mkdir -p "$HF_HOME_PATH"

# Export as environment variables
export RERANKER_MODEL="$MODEL"
export RERANKER_DEVICE="$DEVICE"

# Set HuggingFace cache paths (all point to same configurable location)
export HF_HOME="$HF_HOME_PATH"
export TRANSFORMERS_CACHE="$HF_HOME_PATH"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME_PATH"

# Start the FastAPI server
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
