import subprocess
import time
import os
import requests
import re
import sys

MODELS = [
    "BAAI/bge-reranker-base",
    "jinaai/jina-reranker-v2-base-multilingual",
    "BAAI/bge-reranker-v2-m3"
]

ANCHORS_FILE = "/home/daniel/multistage_assist/multistage_assist_anchors.json"
CACHE_FILE = "/home/daniel/multistage_assist/multistage_assist_semantic_cache.json"

# Ensure cache file exists (empty if needed)
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "w") as f:
        f.write('{"entries": []}')

def start_container(model):
    print(f"Starting container for {model}...")
    # Stop any existing
    subprocess.run(["docker", "rm", "-f", "accuracy_bench"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"Starting container for {model}...")
    # Stop any existing
    subprocess.run(["docker", "rm", "-f", "accuracy_bench"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 1. Create container (stopped)
    cmd = [
        "docker", "create", "--name", "accuracy_bench", "--rm",
        "--security-opt", "seccomp=unconfined",
        "-e", "HF_HUB_DISABLE_XET=1",
        "-e", "HF_HUB_ENABLE_HF_TRANSFER=0",
        "-e", "OPENBLAS_NUM_THREADS=1",
        "-e", f"RERANKER_MODEL={model}",
        "-e", "EMBEDDING_MODEL=BAAI/bge-m3",
        "-e", "ANCHORS_FILE=/anchors.json",
        "-e", "USER_CACHE_FILE=/cache.json",
        "-p", "9876:9876",
        "local-reranker", "python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9876"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    
    # 2. Inject files
    print("Injecting files...")
    subprocess.run(["docker", "cp", ANCHORS_FILE, "accuracy_bench:/anchors.json"], check=True)
    subprocess.run(["docker", "cp", CACHE_FILE, "accuracy_bench:/cache.json"], check=True)
    
    # 3. Start container
    print("Starting container...")
    subprocess.run(["docker", "start", "accuracy_bench"], check=True, stdout=subprocess.DEVNULL)

def wait_for_health():
    print("Waiting for health check...", end="")
    for _ in range(30): # 5 mins max (heavy models load slow)
        try:
            r = requests.get("http://localhost:9876/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(" OK!")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(10)
    print(" Timeout!")
    return False

def run_tests():
    print("Running Pytest...")
    env = os.environ.copy()
    env["RERANKER_HOST"] = "localhost"
    
    # Run pytest and capture output
    result = subprocess.run(
        ["pytest", "tests/integration/test_semantic_cache_comprehensive.py"],
        env=env,
        capture_output=True,
        text=True
    )
    
    # Parse summary
    # e.g. "126 passed, 3 failed in 24.5s"
    match = re.search(r"(=+ )?(\d+) passed", result.stdout)
    passed = int(match.group(2)) if match else 0
    
    match_fail = re.search(r"(\d+) failed", result.stdout)
    failed = int(match_fail.group(1)) if match_fail else 0
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    
    return accuracy, passed, failed, total

results = {}

for model in MODELS:
    print(f"\n\n=== BENCHMARKING: {model} ===")
    try:
        start_container(model)
        if wait_for_health():
            acc, p, f, t = run_tests()
            print(f"Result: {acc:.1f}% ({p}/{t} Passed)")
            results[model] = (acc, p, f)
        else:
            print("Failed to start container")
            subprocess.run(["docker", "logs", "accuracy_bench"])
            results[model] = (0, 0, 0)
    except Exception as e:
        print(f"Error: {e}")
        results[model] = (0, 0, 0)
    finally:
        subprocess.run(["docker", "rm", "-f", "accuracy_bench"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\n\n=== FINAL ACCURACY REPORT ===")
for model, (acc, p, f) in results.items():
    print(f"{acc:>6.1f}% : {model} ({p} pass, {f} fail)")
