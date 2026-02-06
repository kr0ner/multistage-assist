import time
import requests
import statistics

URL = "http://192.168.178.2:9876/lookup"
QUERY = "Schalte alle Rolläden im Haus zu"

def benchmark():
    print(f"Benchmarking Semantic Cache at {URL}...")
    print(f"Query: '{QUERY}'")
    
    latencies = []
    
    for i in range(1, 6):
        start = time.perf_counter()
        try:
            resp = requests.post(URL, json={"query": QUERY, "top_k": 3}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Request {i} failed: {e}")
            return

        end = time.perf_counter()
        duration = end - start
        latencies.append(duration)
        
        found = data.get("found", False)
        match_info = f"Hit: {data.get('intent')}" if found else "Miss"
        
        print(f"Run {i}: {duration:.4f}s ({match_info})")
        if i == 1:
            print("... (Pausing 1s) ...")
            time.sleep(1)

    avg = statistics.mean(latencies)
    print("\nResults:")
    print(f"Cold Start (Run 1): {latencies[0]:.4f}s")
    if len(latencies) > 1:
        warm_avg = statistics.mean(latencies[1:])
        print(f"Warm Avg (Run 2-5): {warm_avg:.4f}s")
    
if __name__ == "__main__":
    benchmark()
