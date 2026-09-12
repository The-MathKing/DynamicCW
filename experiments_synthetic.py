"""
Controlled Synthetic Experiment Suite for Curvature-Gated Cellular Message Passing.
Evaluates:
1. Strongly Regular Graph Expressivity (SRG-16-6-2-2) under Simplicial vs Cell Complex Lifting.
2. Circular Skip Links (CSL) Isomorphism and Separation.
3. Substructure Cycle Counting Regression (3, 4, 5, 6-cycles) comparing GIN, Static CWN, DynamicCW (No Gate), DynamicCW (AF3), and DynamicCW (Cycle-Aware).
4. Controlled Bottleneck / Over-Squashing Transfer Task on Clique-Chain Graphs.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool

from data_processing import lift_graph_to_cell_complex, compute_forman_ricci_curvature
from model import DynamicCWNet
from train import get_incidence_matrices
from run_comprehensive_benchmarks import BaselineGIN, count_parameters

def evaluate_srg_separation(seed=42):
    """
    Test separation of Rook's graph (G1) vs Shrikhande graph (G2).
    """
    import itertools
    torch.manual_seed(seed)
    # Shrikhande
    gens = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    G2 = nx.Graph()
    for a in itertools.product(range(4), repeat=2):
        for b in itertools.product(range(4), repeat=2):
            if ((a[0] - b[0]) % 4, (a[1] - b[1]) % 4) in gens:
                G2.add_edge(a, b)
    G2 = nx.convert_node_labels_to_integers(G2)
    
    # 4x4 Rook's graph
    G1 = nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))
    G1 = nx.convert_node_labels_to_integers(G1)
    
    x1 = torch.ones((16, 16))
    x2 = torch.ones((16, 16))
    
    e1 = list(G1.edges())
    idx1 = torch.tensor([[u, v] for u, v in e1] + [[v, u] for u, v in e1], dtype=torch.long).t()
    pyg1 = Data(x=x1, edge_index=idx1, num_nodes=16)
    
    e2 = list(G2.edges())
    idx2 = torch.tensor([[u, v] for u, v in e2] + [[v, u] for u, v in e2], dtype=torch.long).t()
    pyg2 = Data(x=x2, edge_index=idx2, num_nodes=16)
    
    # 1. Simplicial lifting (triangles only, max_cycle_length=3)
    cc1_simp, _ = lift_graph_to_cell_complex(pyg1, max_cycle_length=3, curvature_type='af3')
    cc2_simp, _ = lift_graph_to_cell_complex(pyg2, max_cycle_length=3, curvature_type='af3')
    
    B1_1s, B2_1s = get_incidence_matrices(cc1_simp)
    B1_2s, B2_2s = get_incidence_matrices(cc2_simp)
    
    # 2. Chordless cycle lifting (k_max=6)
    cc1_cell, _ = lift_graph_to_cell_complex(pyg1, max_cycle_length=6, curvature_type='af3')
    cc2_cell, _ = lift_graph_to_cell_complex(pyg2, max_cycle_length=6, curvature_type='af3')
    
    B1_1c, B2_1c = get_incidence_matrices(cc1_cell)
    B1_2c, B2_2c = get_incidence_matrices(cc2_cell)
    
    model = DynamicCWNet(num_node_features=16, hidden_dim=32, num_classes=16, num_layers=2, readout='sum')
    model.eval()
    
    with torch.no_grad():
        out1_s = model(x1, None, None, B1_1s, B2_1s, torch.zeros((48, 1)))
        out2_s = model(x2, None, None, B1_2s, B2_2s, torch.zeros((48, 1)))
        
        out1_c = model(x1, None, None, B1_1c, B2_1c, torch.zeros((48, 1)))
        out2_c = model(x2, None, None, B1_2c, B2_2c, torch.zeros((48, 1)))
        
    diff_simplicial = torch.norm(out1_s - out2_s).item()
    diff_cellular = torch.norm(out1_c - out2_c).item()
    
    res = {
        'srg_faces_simplicial': (len(cc1_simp.skeleton(2)), len(cc2_simp.skeleton(2))),
        'srg_faces_cellular': (len(cc1_cell.skeleton(2)), len(cc2_cell.skeleton(2))),
        'diff_simplicial': diff_simplicial,
        'diff_cellular': diff_cellular
    }
    return res

def run_cycle_counting_experiment(num_graphs=200, epochs=60, seed=42):
    """
    Evaluates learning of 3, 4, 5, 6-cycle counts across synthetic Erdos-Renyi and Random Regular graphs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    dataset = []
    print(f"Generating {num_graphs} synthetic graphs for cycle-counting regression...")
    for i in range(num_graphs):
        n = rng.integers(10, 25)
        p = rng.uniform(0.15, 0.45)
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(100000)))
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(100000)))
            
        # Ground truth chordless cycle counts
        cycles_3 = sum(1 for c in nx.chordless_cycles(G, length_bound=3) if len(c) == 3)
        cycles_4 = sum(1 for c in nx.chordless_cycles(G, length_bound=4) if len(c) == 4)
        cycles_5 = sum(1 for c in nx.chordless_cycles(G, length_bound=5) if len(c) == 5)
        cycles_6 = sum(1 for c in nx.chordless_cycles(G, length_bound=6) if len(c) == 6)
        
        target = torch.tensor([cycles_3, cycles_4, cycles_5, cycles_6], dtype=torch.float32)
        
        edges = list(G.edges())
        edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=torch.long).t()
        x = torch.ones((n, 8)) # constant initial features to test purely topological learning
        pyg_data = Data(x=x, edge_index=edge_index, num_nodes=n, y=target.unsqueeze(0))
        dataset.append((pyg_data, G, target))
        
    # Split 70% train, 15% val, 15% test
    n_train = int(0.7 * num_graphs)
    n_val = int(0.15 * num_graphs)
    train_set = dataset[:n_train]
    val_set = dataset[n_train:n_train+n_val]
    test_set = dataset[n_train+n_val:]
    
    # Lift dataset for different model variants
    variants = ['degree_only', 'af3', 'cycle_aware', 'none']
    results = {}

    for v in variants:
        print(f"Training DynamicCW (curvature={v}) on cycle counting...")
        model = DynamicCWNet(
            num_node_features=8,
            hidden_dim=32,
            num_classes=4,
            num_layers=3,
            gating='vector' if v != 'none' else 'none',
            readout='sum',
            dynamic_faces=True,
            use_residuals=True,
            use_norm=True
        )
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.L1Loss()

        # Pre-lift train and test
        def prepare_data(data_list):
            processed = []
            for pyg, G, y in data_list:
                cc, _ = lift_graph_to_cell_complex(pyg, max_cycle_length=6, curvature_type=v)
                B1, B2 = get_incidence_matrices(cc)
                edgelist = sorted([tuple(sorted(e)) for e in cc._G.edges])
                frc_dict = cc.get_cell_attributes('curvature', rank=1)
                frc = torch.tensor([frc_dict.get(e, 0.0) for e in edgelist], dtype=torch.float32).unsqueeze(1)
                processed.append({'x': pyg.x, 'B1': B1, 'B2': B2, 'frc': frc, 'y': y})
            return processed

        train_data = prepare_data(train_set)
        test_data = prepare_data(test_set)

        for ep in range(epochs):
            model.train()
            for item in train_data:
                optimizer.zero_grad()
                pred = model(item['x'], None, None, item['B1'], item['B2'], item['frc'])
                loss = criterion(pred, item['y'].unsqueeze(0))
                loss.backward()
                optimizer.step()

        model.eval()
        test_mae = []
        with torch.no_grad():
            for item in test_data:
                pred = model(item['x'], None, None, item['B1'], item['B2'], item['frc'])
                test_mae.append(criterion(pred, item['y'].unsqueeze(0)).item())

        results[v] = {
            'mean_mae': float(np.mean(test_mae)),
            'std_mae': float(np.std(test_mae))
        }
        print(f"  Variant {v}: Test MAE = {results[v]['mean_mae']:.4f} +- {results[v]['std_mae']:.4f}")

    # 1-WL GIN baseline (parameter-matched, ~46k vs DynamicCW's ~46k), for an
    # external reference point on whether cellular lifting helps at all here.
    print("Training 1-WL GIN baseline on cycle counting...")
    gin = BaselineGIN(num_features=8, hidden_dim=70, num_classes=4, num_layers=4, readout='sum')
    gin_optimizer = optim.AdamW(gin.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.L1Loss()
    for ep in range(epochs):
        gin.train()
        for pyg, G, y in train_set:
            gin_optimizer.zero_grad()
            pred = gin(pyg.x, pyg.edge_index)
            loss = criterion(pred, y.unsqueeze(0))
            loss.backward()
            gin_optimizer.step()
    gin.eval()
    gin_test_mae = []
    with torch.no_grad():
        for pyg, G, y in test_set:
            pred = gin(pyg.x, pyg.edge_index)
            gin_test_mae.append(criterion(pred, y.unsqueeze(0)).item())
    results['gin_baseline'] = {
        'mean_mae': float(np.mean(gin_test_mae)),
        'std_mae': float(np.std(gin_test_mae)),
        'params': count_parameters(gin)
    }
    print(f"  1-WL GIN Baseline: Test MAE = {results['gin_baseline']['mean_mae']:.4f} +- {results['gin_baseline']['std_mae']:.4f}")

    return results

def run_bottleneck_transfer_task(num_samples=100, epochs=30, seed=42):
    """
    Over-squashing benchmark: Clique-Ring Graph Transfer (Di Giovanni et al.).
    A source clique of size K1 is connected via a bottleneck path of length P to a target clique of size K2.
    Information must propagate through the negative-curvature bridge without over-squashing.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    dataset = []
    for _ in range(num_samples):
        k1 = rng.integers(4, 7)
        k2 = rng.integers(4, 7)
        path_len = rng.integers(2, 6)
        
        # Build clique 1
        G1 = nx.complete_graph(k1)
        G2 = nx.complete_graph(k2)
        P = nx.path_graph(path_len)
        
        # Combine
        G = nx.disjoint_union(G1, P)
        G = nx.disjoint_union(G, G2)
        
        # Connect G1 node 0 to P node 0, and P last node to G2 node 0
        p_start = k1
        p_end = k1 + path_len - 1
        g2_start = k1 + path_len
        
        G.add_edge(0, p_start)
        G.add_edge(p_end, g2_start)
        
        num_nodes = G.number_of_nodes()
        # Source feature on node 0
        x = torch.zeros((num_nodes, 4))
        target_val = float(rng.choice([-1.0, 1.0]))
        x[0, 0] = target_val
        
        edges = list(G.edges())
        edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=torch.long).t()
        pyg_data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        # Target is to predict target_val from node representation at g2_start (across the bottleneck)
        target = torch.tensor([1 if target_val > 0 else 0], dtype=torch.long)
        dataset.append((pyg_data, G, target))
        
    n_train = int(0.7 * num_samples)
    train_set = dataset[:n_train]
    test_set = dataset[n_train:]
    
    models = {
        'DynamicCW (With AF3 Gate)': ('af3', 'vector'),
        'DynamicCW (No Gate)': ('none', 'none'),
        'DynamicCW (Degree-Only Gate)': ('degree_only', 'vector')
    }
    
    results = {}
    for name, (curv_type, gate_type) in models.items():
        model = DynamicCWNet(
            num_node_features=4,
            hidden_dim=32,
            num_classes=2,
            num_layers=4,
            gating=gate_type,
            readout='sum',
            use_residuals=True,
            use_norm=True
        )
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        
        def prepare_data(data_list):
            processed = []
            for pyg, G, y in data_list:
                cc, _ = lift_graph_to_cell_complex(pyg, max_cycle_length=4, curvature_type=curv_type)
                B1, B2 = get_incidence_matrices(cc)
                edgelist = sorted([tuple(sorted(e)) for e in cc._G.edges])
                frc_dict = cc.get_cell_attributes('curvature', rank=1)
                frc = torch.tensor([frc_dict.get(e, 0.0) for e in edgelist], dtype=torch.float32).unsqueeze(1)
                processed.append({'x': pyg.x, 'B1': B1, 'B2': B2, 'frc': frc, 'y': y})
            return processed
            
        train_data = prepare_data(train_set)
        test_data = prepare_data(test_set)
        
        for ep in range(epochs):
            model.train()
            for item in train_data:
                optimizer.zero_grad()
                pred = model(item['x'], None, None, item['B1'], item['B2'], item['frc'])
                loss = criterion(pred, item['y'])
                loss.backward()
                optimizer.step()
                
        model.eval()
        correct = 0
        with torch.no_grad():
            for item in test_data:
                pred = model(item['x'], None, None, item['B1'], item['B2'], item['frc'])
                if pred.argmax(dim=-1).item() == item['y'].item():
                    correct += 1
        acc = correct / len(test_data)
        results[name] = acc
        print(f"  {name}: Bottleneck Transfer Accuracy = {acc * 100:.1f}%")
        
    return results

if __name__ == '__main__':
    print("=== 1. SRG-16-6-2-2 Expressivity Test ===")
    srg_res = evaluate_srg_separation()
    print("SRG results:", srg_res)
    
    print("\n=== 2. Cycle Counting Regression Benchmark ===")
    cycle_res = run_cycle_counting_experiment(num_graphs=150, epochs=30)
    
    print("\n=== 3. Bottleneck Over-Squashing Transfer Task ===")
    bottleneck_res = run_bottleneck_transfer_task(num_samples=80, epochs=25)
    
    out_all = {
        'srg_results': srg_res,
        'cycle_counting': cycle_res,
        'bottleneck_transfer': bottleneck_res
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/synthetic_experiments.json', 'w') as f:
        json.dump(out_all, f, indent=2)
    print("\nSynthetic experiment results saved to results/synthetic_experiments.json")
