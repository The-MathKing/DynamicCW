import sys, os
sys.path.insert(0, os.path.abspath('..'))
import json
import math
import multiprocessing as mp
import numpy as np
from scipy import stats
from experiments_synthetic import run_cycle_counting_experiment, run_bottleneck_transfer_task

SEEDS = list(range(42, 67))  # 25 seeds
N_PARALLEL_WORKERS = int(os.environ.get('DYNAMICCW_N_WORKERS', '2'))


def _cycle_counting_worker(seed):
    print(f"=== Cycle counting, seed {seed} ===")
    return seed, run_cycle_counting_experiment(num_graphs=150, epochs=60, seed=seed)


def _bottleneck_worker(seed):
    print(f"=== Bottleneck transfer, seed {seed} ===")
    return seed, run_bottleneck_transfer_task(num_samples=150, epochs=30, seed=seed)


def paired_tost(x, y, margin, alpha=0.05):
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    df = n - 1
    if se == 0.0:
        return {'mean_diff': mean_diff, 'se': se, 'p_tost': 0.0 if abs(mean_diff) < margin else 1.0,
                'equivalent_at_05': abs(mean_diff) < margin, 'margin': margin}
    t1 = (mean_diff - (-margin)) / se
    p1 = 1 - stats.t.cdf(t1, df=df)
    t2 = (mean_diff - margin) / se
    p2 = stats.t.cdf(t2, df=df)
    p_tost = float(max(p1, p2))
    return {'mean_diff': mean_diff, 'se': se, 'p_tost': p_tost,
            'equivalent_at_05': bool(p_tost < alpha), 'margin': margin}


def aggregate_cycle_counting():
    per_seed = {}
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=N_PARALLEL_WORKERS) as pool:
        for seed, res in pool.imap_unordered(_cycle_counting_worker, SEEDS):
            for variant, s in res.items():
                per_seed.setdefault(variant, []).append(s['mean_mae'])

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

    # TOST equivalence of each DynamicCW curvature variant against the un-gated ('none') baseline
    if 'none' in per_seed:
        for variant in per_seed:
            if variant in ('none', 'gin_baseline'):
                continue
            tost = paired_tost(per_seed[variant], per_seed['none'], margin=1.0)
            summary[variant]['tost_vs_none'] = tost
    return summary


def aggregate_bottleneck():
    per_seed = {}
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=N_PARALLEL_WORKERS) as pool:
        for seed, res in pool.imap_unordered(_bottleneck_worker, SEEDS):
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

    # TOST equivalence of no-gate / degree-only against the AF3-gated variant (0.10 accuracy margin)
    ref_key = 'DynamicCW (With AF3 Gate)'
    if ref_key in per_seed:
        for name in per_seed:
            if name == ref_key:
                continue
            tost = paired_tost(per_seed[name], per_seed[ref_key], margin=0.10)
            t_stat, p_val = stats.ttest_rel(per_seed[name], per_seed[ref_key])
            summary[name]['paired_ttest_vs_af3'] = {'t_statistic': float(t_stat), 'p_value': float(p_val)}
            summary[name]['tost_vs_af3'] = tost
    return summary


if __name__ == '__main__':
    cycle_summary = aggregate_cycle_counting()
    bottleneck_summary = aggregate_bottleneck()
    os.makedirs('results', exist_ok=True)
    with open('results/synthetic_multiseed_results.json', 'w') as f:
        json.dump({'cycle_counting': cycle_summary, 'bottleneck_transfer': bottleneck_summary, 'num_seeds': len(SEEDS)}, f, indent=2)
    print("Saved results/synthetic_multiseed_results.json")
