---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''
---

## Description
<!-- A clear description of what the bug is. -->


## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
<!-- What did you expect to happen? -->


## Actual Behavior
<!-- What actually happened? -->


## Debug Logs
<!-- IMPORTANT: Please include debug logs for the integration. -->

### How to enable debug logging:
Add this to your `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.multistage_assist: debug
```

Then restart Home Assistant and reproduce the issue.

### Logs:
<!-- Paste the relevant debug logs here. Look for lines containing [Stage0], [Stage1], [SemanticCache], etc. -->

```
PASTE DEBUG LOGS HERE
```

## Environment
- Home Assistant Version: 
- MultiStage Assist Version: 
- Ollama Version: 
- Ollama Model: 
- Python Version: 

## Additional Context
<!-- Any other context about the problem (screenshots, voice assistant used, etc.) -->
