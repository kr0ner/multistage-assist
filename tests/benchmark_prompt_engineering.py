
import requests
import json
import time
import statistics

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:4b-instruct"

# Dataset (Hard Cases)
DATASET = [
    {
        "text": "Wie warm ist es aktuell im Bad?",
        "intent_check": "HassGetState",
        "slots_check": ["area", "device_class"] # Expecting device_class: temperature
    },
    {
        "text": "Mach den Rollo im Schlafzimmer zur Hälfte zu",
        "intent_check": "HassSetPosition",
        "slots_check": ["area", "position"] # Expecting position: 50
    },
    {
        "text": "Setze die Temperatur im Büro auf 21 Grad",
        "intent_check": "HassClimateSetTemperature",
        "slots_check": ["area", "temperature"]
    },
    {
        "text": "Schalte das Licht im Wohnzimmer an und setze die Helligkeit auf 50%",
        "intent_check": "HassTurnOn",
        "slots_check": ["area", "brightness"]
    }
]

# Prompt Variants
PROMPTS = {
    "BASELINE": """You are a smart home assistant. Identify the intent and entities.
Return JSON ONLY.
Schema:
{
  "intent": "HassTurnOn" | "HassTurnOff" | "HassGetState" | ...,
  "slots": {
    "area": "Kitchen",
    "device_class": "light",
     ...
  }
}
""",

    "GERMAN_ROLE": """Du bist ein intelligenter Assistent für Home Assistant.
Deine Aufgabe ist es, Nutzeranfragen in strukturierte Befehle (Intents) zu übersetzen.
Antworte NUR mit JSON.

Schema:
{
  "intent": "IntentName",
  "slots": {
    "area": "Raumname",
    "device_class": "temperature|humidity|illuminance|...",
    "position": 0-100,
    ...
  }
}
""",

    "FEW_SHOT": """You are a smart home assistant. Identify the intent and entities.
Return JSON ONLY.

Examples:
User: "Licht in der Küche an"
JSON: {"intent": "HassTurnOn", "slots": {"area": "Küche", "domain": "light"}}

User: "Wie warm ist es im Wohnzimmer?"
JSON: {"intent": "HassGetState", "slots": {"area": "Wohnzimmer", "device_class": "temperature"}}

User: "Rollo im Bad auf 50%"
JSON: {"intent": "HassSetPosition", "slots": {"area": "Bad", "position": 50, "domain": "cover"}}

User: "Heizung im Büro auf 22 Grad"
JSON: {"intent": "HassClimateSetTemperature", "slots": {"area": "Büro", "temperature": 22}}
""",

    "COT": """You are a smart home assistant.
Think step-by-step before answering.
1. Identify the action (Turn on, set value, query state).
2. Identify the device or area.
3. Identify any values (temperature, position, brightness).
4. Output the result as JSON.

Schema:
{
  "reasoning": "string",
  "intent": "string",
  "slots": {}
}
"""
}

def benchmark_prompt(name, system_prompt):
    print(f"\nTesting Prompt Style: {name}")
    print("-" * 60)
    
    success_count = 0
    latencies = []
    
    for case in DATASET:
        payload = {
            "model": MODEL,
            "prompt": case["text"],
            "system": system_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            start = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            latencies.append(time.time() - start)
            
            if resp.status_code != 200:
                print(f"  [ERR] {resp.status_code}")
                continue
                
            data = resp.json().get("response", "")
            try:
                parsed = json.loads(data)
                intent = parsed.get("intent")
                slots = parsed.get("slots", {})
                
                # Check
                intent_ok = intent == case["intent_check"]
                slots_ok = all(k in slots for k in case["slots_check"])
                
                # Special check for values
                if "position" in case["slots_check"] and slots.get("position") != 50:
                    slots_ok = False
                
                if intent_ok and slots_ok:
                    success_count += 1
                else:
                    print(f"  [X] {case['text'][:20]}... -> {intent}, Slots: {slots}")
                    
            except:
                print(f"  [FAIL] JSON Parse Error")
                
        except Exception as e:
            print(f"  [EX] {e}")

    acc = (success_count / len(DATASET)) * 100
    avg_lat = statistics.mean(latencies) if latencies else 0
    print(f"  -> Accuracy: {acc:.1f}%, Avg Latency: {avg_lat:.2f}s")
    return {"name": name, "accuracy": acc, "latency": avg_lat}

def main():
    results = []
    for name, prompt in PROMPTS.items():
        results.append(benchmark_prompt(name, prompt))
        
    print("\nSUMMARY")
    for r in results:
        print(f"{r['name']:<15} | {r['accuracy']:.1f}% | {r['latency']:.2f}s")

if __name__ == "__main__":
    main()
