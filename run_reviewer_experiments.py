"""
Camera-ready experiments requested by the reviewers (Paper 306).

  zinc_mean     Mean vs. sum aggregation x {no gate, standardized AF3 gate, raw AF3 gate}
                on the same 700-molecule ZINC subsample, split, seeds and training protocol
                as the main ablation grid (Reviewer 1, W2/W4).
  dirichlet     Dirichlet energy of TRAINED depth-6 models (AF3-gated / un-gated x
                with / without residuals+LayerNorm), measured on the ZINC test molecules
                (Reviewer 1, W6).
  bottleneck    Bottleneck transfer task with 100 seeds (was 25) and a shuffled-kappa arm,
                powered for the 84% vs 64% convergence gap (Reviewer 1, W7).

Usage: python run_reviewer_experiments.py [zinc_mean] [dirichlet] [bottleneck]
Results: results/reviewer_*.json
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy import stats

from model import DynamicCWNet
from run_comprehensive_benchmarks import (preprocess_zinc_subset, train_and_eval_model,
                                          count_parameters, paired_tost)

torch.set_num_threads(1)
N_WORKERS = int(os.environ.get('DYNAMICCW_N_WORKERS', '10'))
ZINC_SEEDS = list(range(42, 62))          # identical to the main ablation grid
DIRICHLET_SEEDS = list(range(42, 52))
BOTTLENECK_SEEDS = list(range(42, 142))   # 100 seeds; 42-66 are the original 25

_DATA = {}


def _zinc(curv):
    if curv not in _DATA:
        _DATA[curv] = preprocess_zinc_subset(num_train=500, num_val=100, num_test=100,
                                             curvature_type=curv, max_cycle_length=6)
    return _DATA[curv]


def _base_kwargs(num_node_features, **over):
    kw = {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3,
          'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True,
          'use_norm': True, 'aggregation': 'sum', 'standardize_curvature': True}
    kw.update(over)
    return kw


# ---------------------------------------------------------------- ZINC mean aggregation
ZINC_MEAN_CONFIGS = {
    'Sum-Agg, AF3 Gate (standardized; reference)': dict(),
    'Sum-Agg, AF3 Gate (raw kappa)': dict(standardize_curvature=False),
    'Mean-Agg, No Gate': dict(aggregation='mean', readout='mean', gating='none'),
    'Mean-Agg, AF3 Gate (standardized)': dict(aggregation='mean', readout='mean'),
    'Mean-Agg, AF3 Gate (raw kappa)': dict(aggregation='mean', readout='mean', standardize_curvature=False),
}


def _zinc_worker(args):
    name, seed = args
    train, val, test = _zinc('af3')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DynamicCWNet(**_base_kwargs(train[0]['x_0'].shape[1], **ZINC_MEAN_CONFIGS[name]))
    mae, t = train_and_eval_model(model, train, val, test, epochs=80)
    return name, seed, mae, t, count_parameters(model)


def _compare(a, b, margin):
    t_stat, p = stats.ttest_rel(a, b)
    tost = paired_tost(a, b, margin=margin)
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / math.sqrt(len(d))
    ci90 = [float(d.mean() - stats.t.ppf(0.95, len(d) - 1) * se), float(d.mean() + stats.t.ppf(0.95, len(d) - 1) * se)]
    return {'mean_diff': float(d.mean()), 'ci90': ci90, 'p_value': float(p), 'p_tost': tost['p_tost']}


def run_zinc_mean():
    tasks = [(n, s) for n in ZINC_MEAN_CONFIGS for s in ZINC_SEEDS]
    scores, params = {n: {} for n in ZINC_MEAN_CONFIGS}, {}
    with mp.get_context('spawn').Pool(N_WORKERS) as pool:
        for name, seed, mae, t, n_p in pool.imap_unordered(_zinc_worker, tasks):
            scores[name][seed] = mae
            params[name] = n_p
            print(f"  [zinc_mean] {name} seed {seed}: MAE {mae:.4f} ({t:.0f}s)", flush=True)
    res = {n: {'scores': [scores[n][s] for s in ZINC_SEEDS], 'params': params[n]} for n in ZINC_MEAN_CONFIGS}
    for n in res:
        sc = np.array(res[n]['scores'])
        res[n].update(mean=float(sc.mean()), std=float(sc.std(ddof=1)))
    comparisons = {
        'mean: std gate vs no gate': ('Mean-Agg, AF3 Gate (standardized)', 'Mean-Agg, No Gate'),
        'mean: raw gate vs no gate': ('Mean-Agg, AF3 Gate (raw kappa)', 'Mean-Agg, No Gate'),
        'mean: raw gate vs std gate': ('Mean-Agg, AF3 Gate (raw kappa)', 'Mean-Agg, AF3 Gate (standardized)'),
        'sum: raw gate vs std gate': ('Sum-Agg, AF3 Gate (raw kappa)', 'Sum-Agg, AF3 Gate (standardized; reference)'),
        'mean no gate vs sum reference': ('Mean-Agg, No Gate', 'Sum-Agg, AF3 Gate (standardized; reference)'),
    }
    out = {'configs': res, 'seeds': ZINC_SEEDS,
           'comparisons': {k: _compare(res[a]['scores'], res[b]['scores'], 0.02) for k, (a, b) in comparisons.items()},
           'n_comparisons': len(comparisons)}
    # reproducibility check against the original grid's reference scores
    try:
        grid = json.load(open('results/comprehensive_ablation_results.json'))
        ref = grid['results']['DynamicCW (AF3 Gated, Sum Readout)']['scores']
        out['reference_max_abs_dev_from_original_grid'] = float(np.max(np.abs(np.array(ref) - np.array(res['Sum-Agg, AF3 Gate (standardized; reference)']['scores']))))
    except Exception as e:  # pragma: no cover
        out['reference_check_error'] = str(e)
    json.dump(out, open('results/reviewer_zinc_mean_aggregation.json', 'w'), indent=2)
    for n, r in res.items():
        print(f"{n}: {r['mean']:.4f} +/- {r['std']:.4f} ({r['params']} params)")
    for k, c in out['comparisons'].items():
        print(f"{k}: diff {c['mean_diff']:+.4f} CI90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p={c['p_value']:.4f} p_tost={c['p_tost']:.4f}")


# ---------------------------------------------------------------- trained Dirichlet energy
DIRICHLET_CONFIGS = {
    'AF3-gated, residuals + LayerNorm': dict(gating='vector', use_residuals=True, use_norm=True),
    'AF3-gated, no residuals/LayerNorm': dict(gating='vector', use_residuals=False, use_norm=False),
    'Un-gated, residuals + LayerNorm': dict(gating='none', use_residuals=True, use_norm=True),
    'Un-gated, no residuals/LayerNorm': dict(gating='none', use_residuals=False, use_norm=False),
}


def _layer_energies(model, data):
    from run_dirichlet_energy import compute_normalized_dirichlet_energies
    e0s, e1s = [], []
    model.eval()
    with torch.no_grad():
        for item in data:
            B1, B2 = item['B1'], item['B2']
            nE = B1.shape[1]
            nF = B2.shape[1] if B2 is not None and B2.dim() == 2 else 0
            h0 = model.node_embedding(item['x_0'])
            h1 = model.edge_embedding(torch.zeros((nE, model.edge_embedding.in_features)))
            h2 = model.face_embedding(torch.ones((nF, model.face_embedding.in_features)))
            B2d = B2.to_dense() if (B2 is not None and B2.is_sparse) else B2
            traj0, traj1 = [], []
            e0, e1 = compute_normalized_dirichlet_energies(h0, h1, B1, B2d if nF > 0 else None)
            traj0.append(e0); traj1.append(e1)
            for conv in model.convs:
                h0, h1, h2 = conv(h0, h1, h2, B1, B2, item['frc'])
                e0, e1 = compute_normalized_dirichlet_energies(h0, h1, B1, B2d if nF > 0 else None)
                traj0.append(e0); traj1.append(e1)
            e0s.append(traj0); e1s.append(traj1)
    return np.mean(e0s, axis=0).tolist(), np.mean(e1s, axis=0).tolist()


def _dirichlet_worker(args):
    name, seed = args
    train, val, test = _zinc('af3')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DynamicCWNet(**_base_kwargs(train[0]['x_0'].shape[1], num_layers=6, **DIRICHLET_CONFIGS[name]))
    init_e0, init_e1 = _layer_energies(model, test)
    mae, t = train_and_eval_model(model, train, val, test, epochs=80)
    e0, e1 = _layer_energies(model, test)
    return name, seed, mae, t, init_e0, init_e1, e0, e1


def run_dirichlet():
    tasks = [(n, s) for n in DIRICHLET_CONFIGS for s in DIRICHLET_SEEDS]
    per = {n: [] for n in DIRICHLET_CONFIGS}
    with mp.get_context('spawn').Pool(N_WORKERS) as pool:
        for name, seed, mae, t, ie0, ie1, e0, e1 in pool.imap_unordered(_dirichlet_worker, tasks):
            per[name].append({'seed': seed, 'test_mae': mae, 'init_e0': ie0, 'init_e1': ie1, 'e0': e0, 'e1': e1})
            print(f"  [dirichlet] {name} seed {seed}: MAE {mae:.4f}, trained E0 last layer {e0[-1]:.4f} ({t:.0f}s)", flush=True)
    out = {}
    for n, runs in per.items():
        runs.sort(key=lambda r: r['seed'])
        agg = {}
        for key in ['init_e0', 'init_e1', 'e0', 'e1']:
            arr = np.array([r[key] for r in runs])
            agg[f'mean_{key}'] = arr.mean(0).tolist()
            agg[f'std_{key}'] = arr.std(0).tolist()
        maes = np.array([r['test_mae'] for r in runs])
        agg.update(test_mae_mean=float(maes.mean()), test_mae_std=float(maes.std(ddof=1)), runs=runs)
        out[n] = agg
        print(f"{n}: MAE {maes.mean():.4f}; trained E0 by layer {np.round(agg['mean_e0'], 4).tolist()}")
    json.dump({'configs': out, 'seeds': DIRICHLET_SEEDS, 'num_layers': 6,
               'note': 'energies averaged over the 100 ZINC test molecules, then mean/std over seeds'},
              open('results/reviewer_trained_dirichlet.json', 'w'), indent=2)


# ---------------------------------------------------------------- bottleneck, 100 seeds
def _bottleneck_worker(seed):
    from experiments_synthetic import run_bottleneck_transfer_task
    return seed, run_bottleneck_transfer_task(num_samples=150, epochs=30, seed=seed, include_shuffled=True)


def _mcnemar_exact(a, b):
    """Exact (binomial) McNemar test on paired binary outcomes."""
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {'a_only': n01, 'b_only': n10, 'p_value': p}


def run_bottleneck():
    res = {}
    with mp.get_context('spawn').Pool(N_WORKERS) as pool:
        for seed, r in pool.imap_unordered(_bottleneck_worker, BOTTLENECK_SEEDS):
            res[seed] = r
            print(f"  [bottleneck] seed {seed}: " + ", ".join(f"{k.split('(')[1][:-1]} {v:.2f}" for k, v in r.items()), flush=True)
    names = list(res[BOTTLENECK_SEEDS[0]].keys())
    acc = {n: np.array([res[s][n] for s in BOTTLENECK_SEEDS]) for n in names}
    conv = {n: acc[n] >= 0.9 for n in names}
    ref = 'DynamicCW (With AF3 Gate)'
    out = {'seeds': BOTTLENECK_SEEDS, 'converged_threshold': 0.9, 'arms': {}, 'vs_af3': {}}
    for n in names:
        out['arms'][n] = {'per_seed_acc': acc[n].tolist(), 'converged': int(conv[n].sum()),
                          'n': len(BOTTLENECK_SEEDS), 'rate': float(conv[n].mean()),
                          'converged_first25': int(conv[n][:25].sum())}
        print(f"{n}: converged {int(conv[n].sum())}/{len(BOTTLENECK_SEEDS)} (first 25 seeds: {int(conv[n][:25].sum())}/25)")
    for n in names:
        if n == ref:
            continue
        table = [[int(conv[ref].sum()), int((~conv[ref]).sum())], [int(conv[n].sum()), int((~conv[n]).sum())]]
        _, p_f = stats.fisher_exact(table)
        out['vs_af3'][n] = {'fisher_p': float(p_f), 'mcnemar_exact': _mcnemar_exact(conv[ref], conv[n])}
        print(f"  AF3 vs {n}: Fisher p={p_f:.4f}, exact McNemar p={out['vs_af3'][n]['mcnemar_exact']['p_value']:.4f}")
    json.dump(out, open('results/reviewer_bottleneck_100seeds.json', 'w'), indent=2)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('results', exist_ok=True)
    which = sys.argv[1:] or ['zinc_mean', 'dirichlet', 'bottleneck']
    for w in which:
        t0 = time.time()
        print(f"\n===== {w} =====", flush=True)
        {'zinc_mean': run_zinc_mean, 'dirichlet': run_dirichlet, 'bottleneck': run_bottleneck}[w]()
        print(f"===== {w} done in {(time.time() - t0) / 60:.1f} min =====", flush=True)
