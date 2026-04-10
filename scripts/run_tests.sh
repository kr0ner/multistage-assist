#!/usr/bin/env bash
# ==============================================================================
# run_tests.sh — Test runner for multistage_assist
#
# Usage:
#   ./scripts/run_tests.sh              # Run unit tests only
#   ./scripts/run_tests.sh --all        # Run unit + integration tests
#   ./scripts/run_tests.sh --integration # Run integration tests only
#   ./scripts/run_tests.sh <pytest args> # Pass custom args to pytest
#
# Environment (auto-configured, override if needed):
#   OLLAMA_HOST   — Ollama server IP       (default: 127.0.0.1)
#   OLLAMA_PORT   — Ollama server port     (default: 11434)
#   OLLAMA_MODEL  — Model for Stage 2 LLM  (default: qwen3:4b-q4_K_M)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PYTEST="$VENV_DIR/bin/pytest"

# ----- Ensure venv exists with required deps -----
if [[ ! -f "$PYTHON" ]]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON" -c "import pytest" 2>/dev/null; then
    echo "Installing test dependencies ..."
    "$VENV_DIR/bin/pip" install -q \
        pytest pytest-asyncio aiohttp voluptuous rapidfuzz \
        rank-bm25 hassil pyyaml numpy unicode-rbnf
fi

# ----- Environment defaults -----
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1}"
export OLLAMA_PORT="${OLLAMA_PORT:-11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b-q4_K_M}"

cd "$PROJECT_DIR"

# ----- Parse mode -----
MODE="unit"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --all)         MODE="all" ;;
        --integration) MODE="integration" ;;
        *)             EXTRA_ARGS+=("$arg") ;;
    esac
done

# ----- Run -----
case "$MODE" in
    unit)
        echo "Running unit tests (no external deps required) ..."
        "$PYTHON" -m pytest tests/ --ignore=tests/integration \
            -v --tb=short "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;
    integration)
        echo "Running integration tests (requires Ollama at $OLLAMA_HOST:$OLLAMA_PORT) ..."
        "$PYTHON" -m pytest tests/integration/ \
            -v --tb=short "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;
    all)
        echo "Running ALL tests (requires Ollama at $OLLAMA_HOST:$OLLAMA_PORT) ..."
        "$PYTHON" -m pytest tests/ \
            -v --tb=short "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
        ;;
esac
