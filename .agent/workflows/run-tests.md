---
description: How to run tests for multistage_assist with local Ollama and reranker
---

# Running Tests

## Prerequisites

The test suite uses environment variables for Ollama configuration.

## Environment Variables

Set these before running integration tests:

```bash
export OLLAMA_HOST=127.0.0.1       # Ollama server IP (default: 127.0.0.1)
export OLLAMA_PORT=11434            # Ollama server port (default: 11434)
export OLLAMA_MODEL=qwen3:4b-q4_K_M  # Model to use
```

## Running All Tests

```bash
cd /home/daniel/multistage_assist

# Set environment for local Ollama
export OLLAMA_HOST=127.0.0.1

// turbo
python3 -m pytest tests/ -v --tb=short
```

## Running Only Unit Tests (no external dependencies)

```bash
// turbo
python3 -m pytest tests/test_embedding_principles.py tests/test_time_normalization.py tests/test_cache_ambiguity.py tests/test_cache_skip_logic.py tests/test_entity_keywords.py tests/test_yes_no_detection.py tests/test_nlu_passthrough.py tests/test_messages_de.py -v
```

## Running Integration Tests Only

```bash
// turbo
python3 -m pytest tests/integration/ -v --tb=short
```

## Quick One-Liner (with env vars)

```bash
OLLAMA_HOST=127.0.0.1 python3 -m pytest tests/ -v --tb=short
```

