"""
Integration test configuration.

These tests require a running Ollama server and will make real LLM calls.
Run with: pytest tests/integration/ -v -m integration

Environment variables:
    OLLAMA_HOST: Ollama server IP address (default: 127.0.0.1)
    OLLAMA_PORT: Ollama server port (default: 11434)
    OLLAMA_MODEL: Model to use (default: qwen3:4b-instruct)
"""

import os

# Integration test Ollama configuration from environment
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")


def get_llm_config():
    """Get LLM configuration for integration tests.
    
    Returns dict suitable for passing to capability constructors.
    """
    return {
        "stage1_ip": OLLAMA_HOST,
        "stage1_port": OLLAMA_PORT,
        "stage1_model": OLLAMA_MODEL,
    }
