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

---
**Note**: These principles ensure that the semantic cache remains accurate and does not execute dangerous or unintended actions based on overly aggressive text simplification.

# Verification Strategy

To ensure these principles remain stable over time, every major update MUST pass the **Local Full-Stack Verification** workflow:

1. **Local Fine-Tuning**: Run `tools/fine_tune_cache.py --model minilm --epochs 4` to align the embedding model with the newest training data (which includes Principle 7 number centroids and multi-command escalation).
2. **Mock Add-on Simulation**: Deploy `tools/mock_addon.py` to serve `/embed` and `/lookup` endpoints locally using the fine-tuned model.
3. **Integration Regression**: Run `pytest tests/integration/test_anchor_hit_rate.py` against the local mock endpoint.
4. **Hit Rate Target**: A successful update MUST achieve **> 98% hit rate** on the anchor test suite before being considered stable for production deployment.
