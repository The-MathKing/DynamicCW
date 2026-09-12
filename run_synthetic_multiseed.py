import sys, os
sys.path.insert(0, os.path.abspath('..'))
import json
import numpy as np
from experiments_synthetic import run_cycle_counting_experiment, run_bottleneck_transfer_task

SEEDS = [42, 43, 44, 45, 46]

def aggregate_cycle_counting():
    per_seed = {}
    for seed in SEEDS:
        print(f"=== Cycle counting, seed {seed} ===")
        res = run_cycle_counting_experiment(num_graphs=150, epochs=60, seed=seed)
        for variant, stats in res.items():
            per_seed.setdefault(variant, []).append(stats['mean_mae'])

    summary = {}
    for variant, maes in per_seed.items():
        maes = np.array(maes)
        summary[variant] = {
            'per_seed_mean_mae': maes.tolist(),
            'mean': float(maes.mean()),
            'std': float(maes.std()),
            'ci95': float(1.96 * maes.std() / np.sqrt(len(maes))),
        }
        print(f"{variant}: {maes.mean():.4f} +/- {maes.std():.4f} (95% CI +/- {summary[variant]['ci95']:.4f})")
    return summary

def aggregate_bottleneck():
    per_seed = {}
    for seed in SEEDS:
        print(f"=== Bottleneck transfer, seed {seed} ===")
        res = run_bottleneck_transfer_task(num_samples=80, epochs=30, seed=seed)
        for name, acc in res.items():
            per_seed.setdefault(name, []).append(acc)

    summary = {}
    for name, accs in per_seed.items():
        accs = np.array(accs)
        summary[name] = {
            'per_seed_acc': accs.tolist(),
            'mean': float(accs.mean()),
            'std': float(accs.std()),
        }
        print(f"{name}: {accs.mean()*100:.1f}% +/- {accs.std()*100:.1f}%")
    return summary

if __name__ == '__main__':
    cycle_summary = aggregate_cycle_counting()
    bottleneck_summary = aggregate_bottleneck()
    os.makedirs('results', exist_ok=True)
    with open('../results/synthetic_multiseed_results.json', 'w') as f:
        json.dump({'cycle_counting': cycle_summary, 'bottleneck_transfer': bottleneck_summary}, f, indent=2)
    print("Saved results/synthetic_multiseed_results.json")
