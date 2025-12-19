# Multi-Stage Assist for Home Assistant

<a href="https://www.buymeacoffee.com/kr0ner" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)

**Multi-Stage Assist** is a highly advanced, local-first (with cloud fallback) conversational agent for Home Assistant. It orchestrates multiple processing stages to provide the speed of standard NLU with the intelligence of LLMs.

📚 **[Full Documentation →](docs/wiki/Home.md)**

## Quick Overview

| Stage | Purpose | Technology |
|-------|---------|------------|
| **Stage 0** | Fast path - instant NLU | Home Assistant built-in |
| **Stage 1** | Smart orchestration | Local LLM (Ollama) |
| **Stage 2** | Chat fallback | Google Gemini |

## Key Features

- **Semantic Command Cache** - Instant replay of learned commands
- **Adaptive Learning** - Remembers your custom room/device names
- **Temporary Controls** - "Turn on light for 10 minutes"
- **Natural German Responses** - Optimized for German language

## Quick Start

### Prerequisites

- Home Assistant 2024.1.0+
- [Ollama](https://ollama.ai) with `qwen3:4b-instruct` and `mxbai-embed-large`
- Google Gemini API Key
- Optional: [Reranker Addon](https://github.com/kr0ner/reranker-addon)

### Installation

[![Install via HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=kr0ner&repository=multistage-assist&category=integration)

1. Add via HACS: `https://github.com/kr0ner/multistage-assist`
2. Restart Home Assistant
3. Add integration: **Settings → Devices & Services → Add Integration**
4. Pull embedding model: `ollama pull mxbai-embed-large`

## Expert Configuration

Power users can add YAML settings to `configuration.yaml`:

```yaml
multistage_assist:
  # Semantic Cache tuning
  reranker_threshold: 0.73
  hybrid_enabled: true
  hybrid_alpha: 0.7
  
  # Low-hardware mode (cache-only, no LLM in Stage1)
  skip_stage1_llm: false
  
  # LLM behavior
  llm_timeout: 30
  llm_max_retries: 2
  
  # Debugging
  debug_cache_hits: false
  debug_llm_prompts: false
```

See [Configuration Reference](docs/wiki/Configuration.md) for all options.

## Documentation

| Topic | Link |
|-------|------|
| Architecture | [Architecture.md](docs/wiki/Architecture.md) |
| Capabilities | [Capabilities-Reference.md](docs/wiki/Capabilities-Reference.md) |
| Configuration | [Configuration.md](docs/wiki/Configuration.md) |
| Development | [Development-Guide.md](docs/wiki/Development-Guide.md) |

## Troubleshooting

Enable debug logging:

```yaml
logger:
  logs:
    custom_components.multistage_assist: debug
```

## License

MIT License - see [LICENSE](LICENSE)

**Attribution Required:** [github.com/kr0ner/multistage-assist](https://github.com/kr0ner/multistage-assist)
