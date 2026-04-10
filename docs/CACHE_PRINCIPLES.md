# Semantic Cache Core Principles

To ensure a high-quality semantic cache for German smart home commands, the following 7 principles are non-negotiable. Every future LLM or developer modifying the cache normalization logic MUST adhere to these rules.

## 1. Grammatical Plurality
The system MUST distinguish between **singular** and **plural** forms of entities.
- "Schalte das Licht an" (Singular) ≠ "Schalte die Lichter an" (Plural).
- Normalization should NOT collapse "Lichter" into "Licht".

## 2. Spatial Distinction (Areas and Floors)
The system MUST distinguish different locations, inclusive of floors and specific rooms.
- "im Keller" must be distinct from "im Erdgeschoss".

## 3. Intent Differentiation
Every distinct intent MUST end up in a different part of the vector space.
- "Schalte an" MUST NEVER hit a "Schalte aus" entry.
- "Licht" MUST NOT match "to cover" (Rollo).
- The state (an/aus) is a critical part of the embedding.

## 4. Domain Representation
Domains (Light, Cover, Climate, etc.) must remain distinct and representable.
- They MUST NOT be stripped out of the command during normalization, as this would collapse the vector space to the action alone.

## 5. Logical Opposites
Opposite meanings MUST be distinguishable in the vector space.
- A command to open a blind should not be semantically identical to closing it.

## 6. Multi-Command Escalation (Separators)
Commands containing multiple instructions separated by "und" or commas must NEVER hit a single-command cache entry.
- Any string containing "und" or "," must be mapped to a dedicated escalation token `[MULTIPLE_COMMANDS_ESCALATION]`.
- This ensures that complex commands always escalate to the LLM (Stage 2 & 3) for proper decomposition rather than executing a partial or incorrect cached command.

## 7. Number Normalization (Centroids)
Numbers in commands (percentages, degrees, etc.) are normalized to "centroids" to maximize cache hit rate for varying user inputs while preserving the intent.
- "Rollo auf 37%" and "Rollo auf 82%" both normalize to **"Rollo auf 50 Prozent"**.
- This applies to temperature, brightness, position, and durations.

## 8. Multi word entity/area names
The model needs to understand that some area or entity names consist of more than one word e.g. "Kinder Badezimmer" "Kinderzimmer 1" "Wohn-Esszimmer"

---
**Note**: These principles ensure that the semantic cache remains accurate and does not execute dangerous or unintended actions based on overly aggressive text simplification.

# Verification Strategy

To ensure these principles remain stable over time, run the embedding principles test suite:

```bash
pytest tests/test_embedding_principles.py -v
```

This test file encodes all 7 principles as automated assertions against the normalization pipeline. Every change to cache normalization or embedding logic MUST pass this suite before merging.
