# Multi-Stage Assist for Home Assistant

If you like the project, help me code all night 😴

<a href="https://www.buymeacoffee.com/kr0ner" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

**Multi-Stage Assist** is a highly advanced, local-first (with cloud fallback) conversational agent for Home Assistant. It orchestrates multiple processing stages to provide the speed of standard NLU with the intelligence of LLMs, enabling complex intent recognition, interactive disambiguation, and learning capabilities.

## 🚀 Features

* **Multi-Stage Pipeline:**
    * **Stage 0 (Fast):** Uses Home Assistant's built-in NLU for instant execution of exact commands.
    * **Stage 1 (Smart):** Uses a local LLM (Ollama) for complex intent parsing, fuzzy entity resolution, and clarification of indirect commands (e.g., "It's too dark" → "Turn on lights").
    * **Stage 2 (Chat):** Falls back to Google Gemini for general knowledge and chit-chat if no smart home intent is found.
* **Semantic Command Cache:** Stores successful commands as embeddings for instant replay of similar requests without LLM calls.
* **Adaptive Learning (Memory):** The system learns from your interactions. If you use a new name for a room or device (e.g., "Bad" for "Badezimmer"), it asks for confirmation and remembers it forever.
* **Interactive Disambiguation:** If a command is ambiguous (e.g., "Turn on the light" in a room with three lights), it asks clarifying questions.
* **Context-Aware Execution:**
    * **Indirect Commands:** Understands "It's too dark/bright".
    * **Temporary Controls:** Turn on devices for a duration (e.g., "für 10 Minuten").
    * **Timers:** Dedicated logic for setting timers on specific mobile devices.
    * **Vacuums:** Specialized logic for cleaning specific rooms, floors, or the whole house.
* **Natural Responses:** Generates varied, natural-sounding German confirmation messages.

## 🏗 Architecture

The agent processes every utterance through a sequence of **Stages**:

### 1. Stage 0: The Fast Path (Native NLU)
* **Goal:** Speed.
* **Logic:** Runs a "dry run" of Home Assistant's native intent recognition.
* **Action:** If a single, unambiguous entity is matched, it executes immediately. If results are ambiguous or missing, it **escalates** to Stage 1.

### 2. Stage 1: The Smart Orchestrator (Local LLM - Ollama)
* **Goal:** Intelligence & Control.
* **Capabilities:**
    * **Semantic Cache:** Checks if a similar command was executed before (uses Ollama embeddings).
    * **Clarification:** Rewrites complex inputs (e.g., splits "Turn on light and close blinds" into atomic commands).
    * **Keyword Intent:** Identifies domains/intents even from vague phrasing.
    * **Entity Resolution:** Uses fuzzy matching, area aliases, and "all entities" fallback logic.
    * **Memory:** Checks a local JSON store for learned aliases before asking the LLM.
    * **Command Processor:** Handles the execution flow (Filter by state -> Check Plural -> Disambiguate -> Execute -> Confirm).
* **Action:** Executes the command or asks the user for more info. If it can't determine a command, it **escalates** to Stage 2.

### 3. Stage 2: The Chat Fallback (Google Gemini)
* **Goal:** Conversation.
* **Logic:** Handles open-ended queries or "chit-chat" that isn't related to controlling the house.
* **Features:** Maintains "Sticky Chat" mode—once you start chatting, it stays in chat mode until the session ends.

## 🛠 Prerequisites

1.  **Home Assistant** (tested on recent versions).
2.  **Ollama** running locally (or accessible via network).
    * Recommended Model: `qwen3:4b-instruct` (fast and capable).
    * Embedding Model: `mxbai-embed-large` (for semantic cache, multilingual support).
3.  **Google Gemini API Key** (for Stage 2 chat).

## 📥 Installation

1.  Copy the `multistage_assist` folder to your Home Assistant `custom_components` directory.
2.  Restart Home Assistant.
3.  Go to **Settings > Devices & Services > Add Integration**.
4.  Search for **Multi-Stage Assist**.
5.  Pull the embedding model on your Ollama server:
    ```bash
    ollama pull mxbai-embed-large
    ```

## ⚙️ Configuration

During setup (or via "Configure"), provide:

### Stage 1 (Local Control)
* **IP:** IP address of your Ollama instance (e.g., `127.0.0.1` or `192.168.1.x`).
* **Port:** Default `11434`.
* **Model:** `qwen3:4b-instruct` (or your preferred local model).

### Stage 2 (Chat)
* **Google API Key:** Your Gemini API Key.
* **Model:** `gemini-2.5-flash` (or `gemini-2.0-flash`).

### Embedding (Semantic Cache)
* **IP:** Defaults to Stage 1 IP. Can point to different Ollama server.
* **Port:** Defaults to Stage 1 port.
* **Model:** `mxbai-embed-large` (recommended for German support).

## 🧠 Capabilities

The system is built on modular **Capabilities**:

| Capability | Description |
| :--- | :--- |
| **SemanticCache** | Caches verified commands as embeddings for instant replay. |
| **Clarification** | Splits compound commands ("AND") and translates indirect speech ("too dark"). |
| **KeywordIntent** | Extracts specific slots (brightness, duration) and intents using LLM logic. |
| **EntityResolver** | Finds devices using fuzzy matching, area filters, and device classes. |
| **AreaAlias** | Maps fuzzy names ("Unten", "Keller") to real HA Areas/Floors. |
| **Memory** | Persists learned aliases for Areas and Entities to disk. |
| **Timer** | Specialized flow for setting Android timers via `notify.mobile_app`. |
| **Vacuum** | Specialized flow for `HassVacuumStart` to clean rooms/floors. |
| **CommandProcessor** | The engine that runs the execution pipeline (filters, disambiguation, etc). |

## ✅ Usage Examples

* **Direct Control:** *"Schalte das Licht im Büro an"*
* **Indirect:** *"Im Wohnzimmer ist es zu dunkel"* (Turns on light)
* **Temporary Control:** *"Licht im Bad für 10 Minuten an"*
* **Timer:** *"Stelle einen Timer für 5 Minuten auf Daniels Handy"*
* **Vacuum:** *"Wische das Erdgeschoss"*
* **Learning:**
    * *User:* "Schalte das Spiegellicht an"
    * *System:* "Meinst du 'Badezimmer Spiegel'?"
    * *User:* "Ja"
    * *System:* "Alles klar. Soll ich mir merken, dass 'Spiegellicht' 'Badezimmer Spiegel' bedeutet?"
    * *User:* "Ja" (Saved forever!)

## 📝 TODOs

* [x] **Semantic Cache:** Store successful commands for fast replay.
* [ ] **RAG / Knowledge:** Implement a vector store to query Home Assistant history or documentation.
* [ ] **Refined Timer Learning:** Better flow for learning device nicknames during timer setting.
* [ ] **Visual Feedback:** Add dashboard cards for active clarifications.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Attribution Required:** When using, modifying, or distributing this software, please include a reference to the original repository: [github.com/kr0ner/multistage-assist](https://github.com/kr0ner/multistage-assist)
