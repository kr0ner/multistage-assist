# Troubleshooting

## LLM Connection

### Cannot connect to Ollama

```
Cannot connect to host 192.168.178.108:11434
```

**Solutions:**
1. Verify Ollama is running: `curl http://HOST:11434/api/tags`
2. Check firewall allows port 11434
3. Verify OLLAMA_HOST configuration

---

## Entity Issues

### Entity not found

**Solutions:**
1. Check entity is exposed to conversation assistant
2. Verify area assignment in HA
3. Try using exact device friendly name

### Wrong entity selected

**Solutions:**
1. Use more specific names
2. Add area qualifier ("Licht im Bad")

---

## Timebox Issues

### Invalid slug error

```
invalid slug snapshot_01KCCT4920GMN78CAN9PWY92CT
```

**Cause:** ULID contains uppercase, HA slugs require lowercase.

**Solution:** Update to latest version (passes lowercase UUID).

### Script not found

**Solutions:**
1. Copy `scripts/timebox_entity_state.yaml` to HA config
2. Reload scripts: Developer Tools → YAML → Reload Scripts

---

## Calendar Issues

### Date not resolved

**Cause:** Relative date pattern not recognized.

**Solution:** Update to latest version with support for "in X Tagen".

---

## Debug Logging

```yaml
logger:
  default: warning
  logs:
    custom_components.multistage_assist: debug
```

### Key Log Points

| Logger | Shows |
|--------|-------|
| `stage0` | NLU results |
| `stage1` | LLM flow |
| `clarification` | Command splitting |
| `keyword_intent` | Intent extraction |
| `entity_resolver` | Entity matching |
| `intent_executor` | Execution |
