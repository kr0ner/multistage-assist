# Semantic Reranker Addon

Home Assistant addon that provides a CrossEncoder reranker API for the Multi-Stage Assist semantic cache.

## Installation

1. Add this repository to your Home Assistant addon store
2. Install "Semantic Reranker"
3. Configure and start

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `BAAI/bge-reranker-v2-m3` | CrossEncoder model from HuggingFace |
| `device` | `cpu` | Device to run on (`cpu` or `cuda`) |
| `port` | `8765` | API port |

## API

### Health Check
```
GET /health
```

### Rerank
```
POST /rerank
{
  "query": "Turn on the kitchen light",
  "candidates": [
    "Switch on the lamp in kitchen",
    "Turn off bedroom light"
  ]
}
```

Response:
```json
{
  "scores": [0.89, 0.32],
  "best_index": 0,
  "best_score": 0.89
}
```

## Multi-Stage Assist Configuration

In your Multi-Stage Assist config, set:
```yaml
reranker_ip: "localhost"  # or addon hostname
reranker_port: 8765
reranker_enabled: true
```
