# Multi-Stage Assist: Architecture & Principles

## 1. Multi-Stage Pipeline Philosophy
The system follows a "minimalist escalate" pattern. We prefer the cheapest/fastest path that solves the user's intent with high confidence.

### Stage 1: Fast-Path (Semantic Cache)
- **Goal**: Immediate resolution for common/repeated commands.
- **Anchors**: Minimalist set of ~2 templates per intent/domain pair (e.g., 2 for `HassTurnOn light`, 2 for `HassTurnOn switch`).
- **Normalisation**: Inputs are article-stripped and collapsed into "centroids" (e.g., `helligkeit_einstellen`, `position_einstellen`) to maximize clustering density.
- **Model**: strictly `models/multilingual-minilm`.

### Stage 2: Mid-Path (Local LLM)
- **Goal**: Intent parsing for commands not in cache using local compute.
- **Model**: configurable (default `qwen3.5:4b-q4_K_M`).
- **Logic**: derived intents from linguistic structure.

### Stage 3: Cloud-Path (Reasoning & Chat)
- **Goal**: Complex queries, multi-device management, and general chat.
- **Providers**: Polymorphic (Gemini, Grok, OpenAI, Anthropic).
- **Tool Access**: Full Model Context Protocol (MCP) access to discover areas, entities, and stored facts.
- **Muscle Memory**: Stage 3 can autonomously store successful resolutions back to Stage 1 via `store_cache_entry`. This allows the system to "learn" user phrasing patterns over time.

## 2. Shared Capabilities
Capabilities are reusable modules shared across stages:
- **Knowledge Graph**: Persistent storage for device dependencies (powered_by) and personal memory (user facts).
- **Area Resolver**: Centralized logic for fuzzy area matching (e.g., "EG" -> "Erdgeschoss").
- **MCP Tool Hub**: Registry of tools exposed to Stage 2 and 3.

## 3. Localization & Strings
- **messages_de.py**: Single source of truth for all German linguistic representations.
- **NO hardcoded German strings** are allowed in the execution or capability layer.

## 4. Hardware Independence
- **Embedding Model**: Fixed to `multilingual-minilm` (same as the Add-on) for consistent vector spaces.
- **Inference**: Configurable via Ollama (Local) or API (Cloud).

## 5. Development Principles
- **Avoid Over-Engineering**: Features should emerge from LLM reasoning rather than complex Python conditionals.
- **Test-Driven Requirements**: Every bug fix or feature must have an integration test covering the scenario.
- **Minimalist Anchors**: Do NOT expand the anchor database to solve misses. Instead, refine the normalization pipeline ("Train the model").

## 6. Semantic Cache Principles
To maintain a high-quality vector space for German smart home commands, the normalization pipeline MUST adhere to the [Semantic Cache Core Principles](docs/CACHE_PRINCIPLES.md). These principles are non-negotiable for anyone modifying Stage 1 normalization or the embedding processing logic.
