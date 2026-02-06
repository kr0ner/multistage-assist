
import requests
import json
import time
import statistics

# Configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODELS = [
    "qwen3:4b-q8_0",
    "qwen3:4b-instruct",
    "qwen3:8b-instruct"
]

def pull_model(model_name):
    print(f"Pulling {model_name}...")
    try:
        # Trigger pull (streaming=False to wait)
        resp = requests.post("http://127.0.0.1:11434/api/pull", json={"name": model_name, "stream": False}, timeout=300)
        if resp.status_code == 200:
            print(f"Successfully pulled {model_name}")
        else:
            print(f"Failed to pull {model_name}: {resp.text}")
    except Exception as e:
        print(f"Error pulling {model_name}: {e}")

# German Benchmark Dataset
# Complex commands involving rooms, devices, values, and ambiguity
DATASET = [
    {
        "text": "Schalte das Licht im Wohnzimmer an und setze die Helligkeit auf 50%",
        "intent_check": "HassTurnOn",
        "slots_check": ["area", "brightness"]
    },
    {
        "text": "Wie warm ist es aktuell im Badezimmer?",
        "intent_check": "HassGetState",
        "slots_check": ["area", "device_class"]
    },
    {
        "text": "Mach den Rollo im Schlafzimmer zur Hälfte zu",
        "intent_check": "HassSetPosition",
        "slots_check": ["area", "position"] # Expecting "half" or 50 logic handling
    },
    {
        "text": "Ich gehe jetzt ins Bett, mach alles aus",
        "intent_check": "HassTurnOff",
        "slots_check": ["area"] # implicit "house" or "everything"
    },
    {
        "text": "Setze die Temperatur im Büro auf 21 Grad",
        "intent_check": "HassClimateSetTemperature",
        "slots_check": ["area", "temperature"]
    }
]

# Domain-Specific Examples (Simulating keyword_intent.py logic)
EXAMPLES = {
    "light": """User: "Licht in der Küche an"
JSON: {"intent": "HassTurnOn", "slots": {"area": "Küche", "domain": "light"}}
User: "Licht im Obergeschoss aus"
JSON: {"intent": "HassTurnOff", "slots": {"floor": "Obergeschoss", "domain": "light"}}
User: "Mach das Licht im Büro heller"
JSON: {"intent": "HassLightSet", "slots": {"area": "Büro", "command": "step_up", "domain": "light"}}""",
    
    "cover": """User: "Rollo im Bad auf 50%"
JSON: {"intent": "HassSetPosition", "slots": {"area": "Bad", "position": 50, "domain": "cover"}}
User: "Mach die Rollläden im Schlafzimmer ganz zu"
JSON: {"intent": "HassTurnOff", "slots": {"area": "Schlafzimmer", "domain": "cover"}}""",

    "climate": """User: "Heizung im Büro auf 22 Grad"
JSON: {"intent": "HassClimateSetTemperature", "slots": {"area": "Büro", "temperature": 22}}
User: "Wie warm ist es im Wohnzimmer?"
JSON: {"intent": "HassGetState", "slots": {"area": "Wohnzimmer", "device_class": "temperature"}}""",

    "sensor": """User: "Wie warm ist es im Bad?"
JSON: {"intent": "HassGetState", "slots": {"area": "Bad", "device_class": "temperature"}}
User: "Wieviel Strom verbraucht der Fernseher?"
JSON: {"intent": "HassGetState", "slots": {"name": "Fernseher", "device_class": "power"}}"""
}

BASE_SYSTEM_PROMPT = """You are a smart home assistant. Identify the intent and entities.
Allowed Slots: area, name, domain, floor, duration, command, device_class, position, temperature, brightness.

Examples:
{examples}

Return JSON ONLY.
"""

def benchmark_model(model_name):
    print(f"\nBenchmarking: {model_name}")
    print("-" * 60)
    
    latencies = []
    tokens_per_sec = []
    success_count = 0
    total_latency = 0
    count = 0

    for item in DATASET:
        # Determine domain for test case (Simulation)
        domain = "light" # default
        txt = item["text"].lower()
        if "rollo" in txt: domain = "cover"
        elif "heizung" in txt or "warm" in txt or "temperatur" in txt: domain = "climate"
        
        # Build Dynamic Prompt
        ex = EXAMPLES.get(domain, EXAMPLES["light"])
        dynamic_prompt = BASE_SYSTEM_PROMPT.format(examples=ex)
        
        payload = {
            "model": model_name,
            "prompt": item["text"],
            "system": dynamic_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            start_time = time.time()
            resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
            end_time = time.time()
            
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text}")
                continue
                
            result = resp.json()
            total_duration = result.get("total_duration", 0) / 1e9 # ns to s
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0) / 1e9
            
            # Metrics
            tps = eval_count / eval_duration if eval_duration > 0 else 0
            latencies.append(total_duration)
            tokens_per_sec.append(tps)
            
            # Accuracy Check
            response_text = result.get("response", "")
            try:
                parsed = json.loads(response_text)
                intent = parsed.get("intent")
                slots = parsed.get("slots", {})
                
                # Check
                intent_ok = intent == item["intent_check"]
                slots_ok = all(k in slots for k in item["slots_check"])
                
                if item.get("slots_check") and "device_class" in item["slots_check"] and slots.get("device_class") == "temperature":
                    # Special check for temp
                    pass
                
                if intent_ok and slots_ok:
                    success_count += 1
                    # print(f"  [✓] {prompt[:30]}... ({total_duration:.2f}s, {tps:.1f} t/s)")
                else:
                    print(f"  [X] {item['text'][:30]}... -> Intent: {intent}, Missing Slots: {[k for k in item['slots_check'] if k not in slots]}")
                    
            except:
                print(f"  [FAIL] JSON Parse Error: {response_text[:50]}...")
                
        except Exception as e:
            print(f"  [ERR] Exception: {e}")

    avg_latency = statistics.mean(latencies) if latencies else 0
    avg_tps = statistics.mean(tokens_per_sec) if tokens_per_sec else 0
    accuracy = (success_count / len(DATASET)) * 100
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "tps": avg_tps
    }

def main():
    print("Starting German LLM Benchmark...")
    print(f"Target URL: {OLLAMA_URL}")
    
    results = []
    for model in MODELS:
        pull_model(model)
        res = benchmark_model(model)
        results.append(res)
        
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Latency':<10} | {'Speed (T/s)':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['model']:<20} | {r['accuracy']:.1f}%      | {r['avg_latency']:.2f}s      | {r['tps']:.1f}")

if __name__ == "__main__":
    main()
