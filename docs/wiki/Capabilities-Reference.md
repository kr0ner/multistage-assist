# Capabilities Reference

Complete reference of all available capabilities in MultiStage Assist.

## Core Capabilities

### semantic_cache

**Purpose:** Fast cache lookup via semantic vectors with hybrid BM25 keyword matching.

**Input:** User input text

**Output:** Cached resolution or None

**Features:**
- **Anchor patterns** - Pre-generated command patterns for all entities/areas
- **3-tier lookup** - Anchor check (fastest) → Local fuzzy (learned entries, top 5) → Remote add-on fallback
- **Hybrid search** - Combines vector similarity + BM25 keyword matching
- **User learning** - Stores verified successful commands
- Preserves disambiguation context for re-prompting

**Skip Filters (commands NOT cached):**
- Short texts (< 3 words) - likely disambiguation responses
- Disambiguation follow-ups ("Küche", "Beide", "das erste")
- Relative brightness commands (`step_up`/`step_down`) - depend on current state
- Timer intents (`HassTimerSet`, `HassTimerCancel`) - context-dependent
- Calendar intents (`HassCalendarCreate`) - unique events
- Delayed control intents - time-sensitive

**Config Options:**
- `vector_search_threshold`: Min score for hit (default: 0.75)
- `hybrid_enabled`: Enable hybrid search (default: true)
- `hybrid_alpha`: Vector vs keyword weight (default: 0.7)
- `cache_max_entries`: Max user cache size (default: 10000)

**Log Messages:**
```
[SemanticCache] HIT (0.92): 'HassTurnOn' -> ['light.kuche']
[SemanticCache] Stored: 'Licht in der Küche an' -> HassTurnOn
[SemanticCache] SKIP too short (2 words): 'Die Spots'
[SemanticCache] SKIP disambig response: 'Küche'
[SemanticCache] SKIP relative command (step_down): 'Mache das Licht dunkler'
[SemanticCache] SKIP timer: 'Stelle einen Timer für 5 Minuten'
```

---

### implicit_intent

**Purpose:** Rephrase vague/implicit commands into explicit ones.

**Input:** User input text

**Output:** List of rephrased commands

**Features:**
- Fast-path direct mappings checked first (no LLM)
- LLM fallback (temperature=0.3) for implicit phrases not in direct mappings

**Implicit Phrases:**
- "zu dunkel" → "Mache Licht heller"
- "zu hell" → "Mache Licht dunkler"
- "zu kalt" → "Mache Heizung wärmer"
- "zu warm" → "Mache Heizung kälter"

---

### atomic_command

**Purpose:** Split compound commands into individual atomic commands.

**Input:** User input text

**Output:** Array of atomic commands

**Features:**
- Splits on "und", "dann", commas
- Multi-area detection ("Licht in der Küche an und im Flur aus")
- Floor splitting ("Erdgeschoss und Obergeschoss")
- LLM temperature: 0.1 (very deterministic)

**Example:**
```
Input: "Licht an und Rollo runter"
Output: ["Schalte Licht an", "Fahre Rollo runter"]
```

---

### keyword_intent

**Purpose:** Detect domain and extract intent from keywords.

**Input:** User input text

**Output:** `{domain, intent, slots}`

**Supported Domains:**
- light, cover, switch, fan
- climate, sensor, media_player
- timer, vacuum, calendar, automation

**Example:**
```
Input: "Licht im Bad auf 50%"
Output: {
  domain: "light",
  intent: "HassLightSet",
  slots: {area: "Bad", brightness: "50"}
}
```

---

### intent_resolution

**Purpose:** Resolve entities from slots using fuzzy matching.

**Input:** User input, keyword_intent data

**Output:** `{intent, slots, entity_ids, learning_data}`

**Features:**
- Area alias resolution
- Floor detection
- Fuzzy entity name matching

---

### entity_resolver

**Purpose:** Find entity IDs matching criteria.

**Input:** Domain, area, floor, name filters

**Output:** List of entity IDs

**Features:**
- Exposure filtering
- Domain filtering
- Area/floor/name matching

---

### intent_executor

**Purpose:** Execute Home Assistant intents.

**Input:** User input, intent name, entity IDs, params

**Output:** Conversation result

**Features:**
- Brightness step up/down (relative %)
- Timebox calls for temporary controls
- State verification after execution
- Error handling

**Brightness Logic:**
- step_up: +20% of current (min 5%)
- step_down: -20% of current (min 5%)
- From 0%: turns on to 30%

---

### intent_confirmation

**Purpose:** Generate natural language confirmation.

**Input:** Intent, devices, params

**Output:** German confirmation text

**Example:**
```
Intent: HassLightSet
Devices: ["Büro"]
Params: {brightness: 50}
Output: "Das Licht im Büro ist auf 50% gesetzt."
```

---

### disambiguation

**Purpose:** Help user select from ambiguous entities.

**Input:** User input, candidate entities

**Output:** Narrowed candidates or selection

---

### disambiguation_select

**Purpose:** Handle user's disambiguation choice.

**Input:** User response, candidates

**Output:** Selected entity

---

## Domain Capabilities

### calendar

**Purpose:** Create calendar events.

**Input:** User input, intent, slots

**Output:** Event creation result or follow-up question

**Features:**
- Multi-turn conversation
- Relative date resolution
- Time range parsing
- Calendar selection
- Confirmation flow

**Date Patterns:**
- heute, morgen, übermorgen
- in X Tagen
- nächsten Montag
- am 25. Dezember

---

### timer

**Purpose:** Set timers with notifications.

**Input:** User input, intent, slots

**Output:** Timer creation result

**Features:**
- Duration parsing
- Named timers
- Device notifications
- Memory integration for devices

---

## Utility Capabilities

### memory

**Purpose:** Store and retrieve learned aliases and personal data (unified in `KnowledgeGraphCapability`).

**Storage:**
- Area aliases: "bad" → "Badezimmer"
- Entity aliases: "daniels handy" → "notify.mobile_app_..."
- Floor mappings
- Device dependencies: powered_by, coupled_with, lux_sensor, associated_cover, energy_parent
- Personal data: names, preferences, birthdays, roles

---

### area_alias

**Purpose:** Fuzzy match area names using `AreaResolverCapability`.

**Input:** User query, candidate areas

**Output:** Best matching area

**Features:**
- Synonym handling (Bad → Badezimmer)
- Global scope detection (Haus, Wohnung)
- LLM-powered matching

---

### plural_detection

**Purpose:** Detect if user is referring to multiple entities.

**Input:** User text, domain

**Output:** Boolean is_plural

**Examples:**
- "alle Lichter" → plural
- "das Licht" → singular

---

### yes_no_response

**Purpose:** Generate yes/no answers for state queries.

---

### mcp

**Purpose:** MCP tool registry for LLM access to home automation.

**9 Tools Registered:**
| Tool | Purpose |
|---|---|
| `list_areas` | Rooms/areas with floor assignments and aliases |
| `list_entities` | Smart devices filtered by domain/area/device_class |
| `get_entity_details` | State and attributes for specific entity |
| `list_automations` | All exposed automations |
| `get_automation_details` | Config/state of automation |
| `store_personal_data` | Save household facts |
| `get_personal_data` | Retrieve stored facts by key |
| `get_system_capabilities` | List intents, domains, features |
| `store_cache_entry` | Learn command phrasing in semantic cache |

**LLM Reasoning:** `resolve_intent_via_llm()` for multi-turn tool reasoning via OllamaClient.

---

### step_control

**Purpose:** Relative adjustments for brightness and cover position.

**Step Sizes:**
- **Brightness**: ±35% of current value (minimum step: 10%)
- **Cover**: ±25% of current position
- **From off**: step_up turns on to 30%

---

### prompt_context

**Purpose:** Build entity/area context for LLM prompts.

**Input:** Hint list specifying which context to include

**Output:** Formatted context string for system prompts

---

### command_processor

**Purpose:** Orchestrate the full execution pipeline.

**Flow:**
1. State-aware candidate filtering (turn off → only ON entities)
2. Plural detection ("alle" = no disambiguation)
3. Entity disambiguation (if multiple candidates)
4. Intent execution with Knowledge Graph prerequisites
5. Natural language confirmation
6. Semantic cache storage (on verified success)

**Methods:** `process()`, `continue_disambiguation()`, `re_prompt_pending()`

---

### chat

**Purpose:** Free-form conversation fallback.

**Input:** User input, context

**Output:** Conversational response

---

## Capability Configuration

### Prompt Schema

Each LLM-based capability defines:

```python
PROMPT = {
    "system": "System instruction for LLM",
    "schema": {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"}
        }
    }
}
```

### Registration

Capabilities are registered in stages:

```python
# stage1.py
self.register(ClarificationCapability(hass, llm_config))
self.register(KeywordIntentCapability(hass, llm_config))
# ... etc
```

### Usage

```python
# Use capability and get result
result = await self.use("capability_name", user_input)

# Get capability instance
cap = self.get("capability_name")
await cap.run(user_input, **params)
```
