import pytest
import json
import asyncio
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from multistage_assist.ollama_client import OllamaClient
from multistage_assist.const import CONF_STAGE1_IP, CONF_STAGE1_PORT, CONF_STAGE1_MODEL

# Tool definitions to test
TOOLS_DEF = [
    {
        "name": "list_entities",
        "description": "List entities filtered by attributes.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Entity domain (e.g. light, switch)"},
                "area_name": {"type": "string", "description": "Filter by area name (fuzzy)"},
            },
        },
    },
]

@pytest.mark.asyncio
async def test_ollama_tool_calling_live():
    """Integration test to verify Ollama can use tools (Live)."""

    # 1. Config
    ip = "127.0.0.1"
    port = 11434
    model = "qwen3:4b-q4_K_M"

    client = OllamaClient(ip, port)

    # 2. Check Connection
    try:
        models = await client.get_models()
        model_names = [m.get('name') for m in models] if models and isinstance(models[0], dict) else models

        # Check if target model is available (partial match)
        if not any(model in m for m in model_names):
            pytest.skip(f"Model {model} not found in {model_names}")

    except Exception as e:
        pytest.skip(f"Ollama not reachable at {ip}:{port}: {e}")

    # 3. Construct Prompt
    tools_json = json.dumps(TOOLS_DEF, indent=2)
    system_prompt = f"""You are a smart home agent assistant.
Your goal is to find the correct entity ID.

Tools:
{tools_json}

Instructions:
1. Analyze the User Request.
2. Decide which tool to call.
3. OUTPUT FORMAT (Strict JSON):
   - To Call Tool: {{"tool": "tool_name", "args": {{...}}}}
   - To Finish: {{"final_answer": ["entity_id"]}}

Do not output any text outside the JSON.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Request: 'Turn on the kitchen lights'\nIntent: HassTurnOn\nDomain: light\nSlots: {{'area': 'kitchen'}}\n\nResolve the entity ID."}
    ]

    # 4. Run Chat
    print(f"\nSending prompt to {model}...")
    try:
        response = await client.chat_completion(model, messages, temperature=0.0)
        print(f"Response: {response}")

        # 5. Validate JSON
        cleaned = response.strip()
        if "{" in cleaned and "}" in cleaned:
             cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

        data = json.loads(cleaned)

        # 6. Assert Tool Call
        assert "tool" in data, "Response should be a tool call"
        assert data["tool"] == "list_entities"
        assert "area_name" in data.get("args", {})
        assert data["args"]["area_name"] == "kitchen"

    except json.JSONDecodeError:
        pytest.fail(f"LLM did not return valid JSON: {response}")
    except Exception as e:
        pytest.fail(f"Chat completion failed: {e}")
