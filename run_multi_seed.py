import sys
import os
import subprocess
import json
from datetime import datetime

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting multi-seed benchmark run at {timestamp}...")
    
    # Run v2 benchmarks (isomorphism, targeted robustness, latency, transfer)
    print("Running v2 benchmarks (run_all_benchmarks_v2.py)...")
    subprocess.run([sys.executable, "run_all_benchmarks_v2.py"], check=True)
    os.rename("results_full_run_v2.json", os.path.join(results_dir, f"results_v2_{timestamp}.json"))
    
    # Run full benchmarks (diversity, robustness, ablation, scalability, zero-shot)
    # The script internally uses n_trials=10, covering the multi-seed requirement.
    print("Running full benchmarks (run_full_benchmarks.py)...")
    subprocess.run([sys.executable, "run_full_benchmarks.py"], check=True)
    os.rename("results_full_run.json", os.path.join(results_dir, f"results_full_{timestamp}.json"))
    
    print(f"All benchmarks finished! Results saved to {results_dir}")

if __name__ == "__main__":
    main()
