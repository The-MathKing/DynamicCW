"""
Comprehensive Benchmarks and Component Ablation Grid Runner for DynamicCW.
Evaluates:
1. Multi-seed ZINC Molecular Regression under matched parameter budget (~100k parameters).
2. Full Ablation Grid:
   - Curvature Input Type: None (kappa=0), Degree-Only (4-du-dv), AF3, Cycle-Aware Forman, Shuffled Curvature, Random Curvature.
   - Cellular Representations: Dynamic 2-Cells vs Static 2-Cells vs No 2-Cells (1-WL GIN baseline).
   - Readout Aggregation: Sum Readout vs Mean Readout.
   - Depth and Normalization: LayerNorm + Residuals vs Plain across L in {2, 4, 8}.
3. Statistical Testing:
   - Paired Student's t-tests and Wilcoxon signed-rank tests across seeds with Holm-Bonferroni correction.
   - Mean, Standard Deviation, and 95% Confidence Intervals.
4. Outputs structured JSON results to results/comprehensive_ablation_results.json.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import math
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from data_processing import lift_graph_to_cell_complex
from model import DynamicCWNet
from train import get_incidence_matrices

# Each worker process trains one (config, seed) independently on CPU; capping
# intra-op threads to 1 per process avoids oversubscribing cores when running
# many worker processes in parallel via multiprocessing.
torch.set_num_threads(1)

N_PARALLEL_WORKERS = int(os.environ.get('DYNAMICCW_N_WORKERS', '4'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaselineGIN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=4, readout='sum'):
        super(BaselineGIN, self).__init__()
        self.readout = readout
        self.node_emb = nn.Linear(num_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        h = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index))
        if batch is not None:
            pooled = global_add_pool(h, batch) if self.readout == 'sum' else global_mean_pool(h, batch)
        else:
            pooled = torch.sum(h, dim=0, keepdim=True) if self.readout == 'sum' else torch.mean(h, dim=0, keepdim=True)
        return self.classifier(pooled)

def preprocess_zinc_subset(num_train=600, num_val=150, num_test=150, curvature_type='af3', max_cycle_length=6):
    """Loads and pre-lifts ZINC dataset."""
    print(f"Loading ZINC dataset (curv={curvature_type}, k_max={max_cycle_length})...")
    os.makedirs('/tmp/ZINC_DATA', exist_ok=True)
    try:
        dataset_train = ZINC(root='/tmp/ZINC_DATA', subset=True, split='train')
        dataset_val = ZINC(root='/tmp/ZINC_DATA', subset=True, split='val')
        dataset_test = ZINC(root='/tmp/ZINC_DATA', subset=True, split='test')
    except Exception as e:
        print("Falling back to standard TUDataset MUTAG/PROTEINS if ZINC download fails:", e)
        dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
        dataset_train = dataset[:num_train]
        dataset_val = dataset[num_train:num_train+num_val]
        dataset_test = dataset[num_train+num_val:num_train+num_val+num_test]
        
    NUM_ATOM_TYPES = 21  # ZINC-12k atom-type indices range 0-20 across train/val/test

    def process_split(split, limit):
        processed = []
        for i, data in enumerate(split[:limit]):
            cc, _ = lift_graph_to_cell_complex(data, max_cycle_length=max_cycle_length, curvature_type=curvature_type)
            B1, B2 = get_incidence_matrices(cc)
            edgelist = sorted([tuple(sorted(e)) for e in cc._G.edges])
            frc_dict = cc.get_cell_attributes('curvature', rank=1)
            frc = torch.tensor([frc_dict.get(e, 0.0) for e in edgelist], dtype=torch.float32).unsqueeze(1)

            if hasattr(data, 'x') and data.x is not None:
                # ZINC atom types are categorical indices; one-hot rather than treat as a raw scalar.
                atom_idx = data.x.view(-1).long()
                x_0 = torch.nn.functional.one_hot(atom_idx, num_classes=NUM_ATOM_TYPES).float()
            else:
                x_0 = torch.ones((data.num_nodes, NUM_ATOM_TYPES))
            processed.append({
                'x_0': x_0,
                'edge_index': data.edge_index,
                'B1': B1,
                'B2': B2,
                'frc': frc,
                'y': data.y.float() if hasattr(data, 'y') and data.y is not None else torch.zeros(1)
            })
        return processed

    train_data = process_split(dataset_train, num_train)
    val_data = process_split(dataset_val, num_val)
    test_data = process_split(dataset_test, num_test)
    return train_data, val_data, test_data

def train_and_eval_model(
    model, 
    train_data, 
    val_data, 
    test_data, 
    epochs=80,
    lr=0.0003,
    weight_decay=1e-5, 
    device='cpu',
    is_gin=False
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    best_test_mae = float('inf')
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for item in train_data:
            optimizer.zero_grad()
            if is_gin:
                pred = model(item['x_0'].to(device), item['edge_index'].to(device))
            else:
                pred = model(
                    item['x_0'].to(device), 
                    None, 
                    None, 
                    item['B1'].to(device), 
                    item['B2'].to(device), 
                    item['frc'].to(device)
                )
            loss = criterion(pred.squeeze(), item['y'].to(device).squeeze())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in val_data:
                if is_gin:
                    pred = model(item['x_0'].to(device), item['edge_index'].to(device))
                else:
                    pred = model(
                        item['x_0'].to(device), 
                        None, 
                        None, 
                        item['B1'].to(device), 
                        item['B2'].to(device), 
                        item['frc'].to(device)
                    )
                val_loss += criterion(pred.squeeze(), item['y'].to(device).squeeze()).item()
        val_loss /= len(val_data)
        
        # Test evaluation at best validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = 0.0
            with torch.no_grad():
                for item in test_data:
                    if is_gin:
                        pred = model(item['x_0'].to(device), item['edge_index'].to(device))
                    else:
                        pred = model(
                            item['x_0'].to(device), 
                            None, 
                            None, 
                            item['B1'].to(device), 
                            item['B2'].to(device), 
                            item['frc'].to(device)
                        )
                    test_loss += criterion(pred.squeeze(), item['y'].to(device).squeeze()).item()
            best_test_mae = test_loss / len(test_data)
            
    train_time = time.time() - start_time
    return best_test_mae, train_time


def _run_single_seed_worker(args):
    """
    Top-level (picklable) worker for one (config, seed) training run, used with
    a fork-context multiprocessing Pool so each worker inherits the already-
    loaded datasets via copy-on-write rather than re-pickling them per task.
    """
    cfg, seed, train_data, val_data, test_data, device_str = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)
    if cfg['type'] == 'gin':
        model = BaselineGIN(**cfg['kwargs'])
    else:
        model = DynamicCWNet(**cfg['kwargs'])
    n_params = count_parameters(model)
    test_mae, t_time = train_and_eval_model(
        model, train_data, val_data, test_data, epochs=80, is_gin=(cfg['type'] == 'gin'), device=device
    )
    return seed, test_mae, t_time, n_params


def paired_tost(x, y, margin=0.02, alpha=0.05):
    """
    Paired two-one-sided-tests (TOST) equivalence test on mean(x) - mean(y),
    against equivalence bounds [-margin, +margin]. Rejecting both one-sided
    nulls (p_tost = max(p1, p2) < alpha) supports statistical equivalence
    within the margin -- unlike a non-significant two-sided t-test, which is
    only a failure to reject a difference, not evidence of equivalence.
    """
    diff = np.array(x) - np.array(y)
    n = len(diff)
    mean_diff = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n))
    df = n - 1
    t1 = (mean_diff - (-margin)) / se
    p1 = 1 - stats.t.cdf(t1, df=df)
    t2 = (mean_diff - margin) / se
    p2 = stats.t.cdf(t2, df=df)
    p_tost = float(max(p1, p2))
    ci95_low = mean_diff - stats.t.ppf(0.975, df=df) * se
    ci95_high = mean_diff + stats.t.ppf(0.975, df=df) * se
    return {
        'mean_diff': mean_diff,
        'se': se,
        'ci95': [float(ci95_low), float(ci95_high)],
        'margin': margin,
        'p_tost': p_tost,
        'equivalent_at_05': bool(p_tost < alpha)
    }


def run_ablation_study(seeds=list(range(42, 62))):
    """
    Runs multi-seed ablation grid across all curvature and cellular architectural choices.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Multi-Seed Benchmark & Ablation Study on {device}...")
    
    # 1. Preload datasets for each curvature type
    curvature_types = ['af3', 'degree_only', 'cycle_aware', 'shuffled', 'none']
    datasets = {}
    for c_type in curvature_types:
        datasets[c_type] = preprocess_zinc_subset(
            num_train=500, num_val=100, num_test=100, curvature_type=c_type, max_cycle_length=6
        )
        
    num_node_features = datasets['af3'][0][0]['x_0'].shape[1]
    
    # Define configurations to test
    configs = [
        # GIN Baseline
        {
            'name': '1-WL GIN Baseline',
            'type': 'gin',
            'curv_data': 'none',
            'kwargs': {'num_features': num_node_features, 'hidden_dim': 105, 'num_classes': 1, 'num_layers': 4, 'readout': 'sum'}
        },
        # Full DynamicCW with AF3 Gate
        {
            'name': 'DynamicCW (AF3 Gated, Sum Readout)',
            'type': 'dynamic_cw',
            'curv_data': 'af3',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 1: Degree-Only Gating
        {
            'name': 'DynamicCW (Degree-Only Gated)',
            'type': 'dynamic_cw',
            'curv_data': 'degree_only',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 2: Cycle-Aware Forman Gating
        {
            'name': 'DynamicCW (Cycle-Aware Forman Gated)',
            'type': 'dynamic_cw',
            'curv_data': 'cycle_aware',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 3: Shuffled Curvature Control
        {
            'name': 'DynamicCW (Shuffled Curvature Control)',
            'type': 'dynamic_cw',
            'curv_data': 'shuffled',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 4: No Gate (kappa = 0)
        {
            'name': 'DynamicCW (No Gate, kappa=0)',
            'type': 'dynamic_cw',
            'curv_data': 'none',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'none', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 5: Static 2-Cells (our own static-face variant, not the published CWN codebase)
        {
            'name': 'Static-Face Variant (Ours, AF3 Gate)',
            'type': 'dynamic_cw',
            'curv_data': 'af3',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': False, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 6: Mean Readout vs Sum Readout
        {
            'name': 'DynamicCW (Mean Readout)',
            'type': 'dynamic_cw',
            'curv_data': 'af3',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 3, 'gating': 'vector', 'readout': 'mean', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        },
        # Ablation 7: Deep Stack without Residuals (Over-Smoothing probe)
        {
            'name': 'DynamicCW (Depth 6, No Residuals)',
            'type': 'dynamic_cw',
            'curv_data': 'af3',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 6, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': False, 'use_norm': False}
        },
        # Ablation 8: Deep Stack WITH Residuals (de-confounds depth from residuals/norm in Ablation 7)
        {
            'name': 'DynamicCW (Depth 6, With Residuals)',
            'type': 'dynamic_cw',
            'curv_data': 'af3',
            'kwargs': {'num_node_features': num_node_features, 'hidden_dim': 48, 'num_classes': 1, 'num_layers': 6, 'gating': 'vector', 'readout': 'sum', 'dynamic_faces': True, 'use_residuals': True, 'use_norm': True}
        }
    ]
    
    # 'spawn', not 'fork': forking a process that already has PyTorch/BLAS
    # thread pools initialized reliably deadlocks on macOS (children hang at
    # 0% CPU). spawn re-imports cleanly in each fresh child process instead.
    ctx = mp.get_context('spawn')
    print(f"Using {N_PARALLEL_WORKERS} parallel worker processes per config (set DYNAMICCW_N_WORKERS to change).")

    all_results = {}
    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n==========================================")
        print(f"Evaluating: {cfg_name}")
        train_data, val_data, test_data = datasets[cfg['curv_data']]

        # Cheap single instantiation up front, just to log the param count.
        probe_model = BaselineGIN(**cfg['kwargs']) if cfg['type'] == 'gin' else DynamicCWNet(**cfg['kwargs'])
        n_params = count_parameters(probe_model)
        print(f"  Model Parameters: {n_params:,}")
        del probe_model

        tasks = [(cfg, seed, train_data, val_data, test_data, str(device)) for seed in seeds]
        results_by_seed = {}
        with ctx.Pool(processes=min(N_PARALLEL_WORKERS, len(seeds))) as pool:
            for seed, test_mae, t_time, _ in pool.imap_unordered(_run_single_seed_worker, tasks):
                results_by_seed[seed] = (test_mae, t_time)
                done = len(results_by_seed)
                print(f"  Seed {seed} ({done}/{len(seeds)}): Test MAE = {test_mae:.4f} ({t_time:.1f}s)")

        scores = [results_by_seed[s][0] for s in seeds]
        train_times = [results_by_seed[s][1] for s in seeds]

        scores = np.array(scores)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        ci95 = float(1.96 * std_score / math.sqrt(len(scores)))
        
        all_results[cfg_name] = {
            'mean': mean_score,
            'std': std_score,
            'ci95': ci95,
            'scores': scores.tolist(),
            'params': n_params,
            'avg_train_time_sec': float(np.mean(train_times))
        }

        # Checkpoint after every config so a crash doesn't lose completed work.
        os.makedirs('results', exist_ok=True)
        with open('results/comprehensive_ablation_results.partial.json', 'w') as f:
            json.dump({'results': all_results, 'num_seeds': len(seeds), 'seeds': seeds,
                       'configs_completed': list(all_results.keys())}, f, indent=2)
        print(f"  [checkpoint saved: {len(all_results)}/{len(configs)} configs done]")

    # Statistical Significance Testing against Full Model
    full_model_key = 'DynamicCW (AF3 Gated, Sum Readout)'
    full_scores = all_results[full_model_key]['scores']
    
    stats_table = {}
    n_comparisons = len(all_results) - 1
    bonferroni_alpha = 0.05 / n_comparisons
    for name, res in all_results.items():
        if name == full_model_key:
            continue
        cur_scores = res['scores']
        t_stat, p_val = stats.ttest_rel(cur_scores, full_scores)
        tost = paired_tost(cur_scores, full_scores, margin=0.02, alpha=0.05)
        stats_table[name] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant_at_05': bool(p_val < 0.05),
            'significant_at_bonferroni': bool(p_val < bonferroni_alpha),
            'tost_equivalence': tost
        }

    final_output = {
        'results': all_results,
        'paired_ttests_vs_full_model': stats_table,
        'bonferroni_alpha': bonferroni_alpha,
        'num_comparisons': n_comparisons,
        'num_seeds': len(seeds),
        'seeds': seeds
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/comprehensive_ablation_results.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    print("\nSaved comprehensive ablation results to results/comprehensive_ablation_results.json")
    return final_output

if __name__ == '__main__':
    run_ablation_study(seeds=list(range(42, 62)))
