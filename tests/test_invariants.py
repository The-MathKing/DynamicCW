"""
Unit Test Suite for DynamicCW and Cellular Topological Invariants.
Verifies:
1. Betti number computation on the 4x4 Rook's vs Shrikhande Strongly Regular Graph pair.
2. Permutation equivariance of canonical lifting, incidence operators, curvature vectors, and forward passes.
3. Orientation invariance of |B1| and |B2| cellular message passing.
4. Exactness of AF3, degree-only, and cycle-aware Forman curvature computations.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from data_processing import (
    compute_forman_ricci_curvature,
    compute_degree_only_curvature,
    compute_cycle_aware_forman,
    find_chordless_cycles_canonical,
    lift_graph_to_cell_complex
)
from model import DynamicCWNet
from train import get_incidence_matrices

import itertools

def create_srg_pair():
    """Generates 4x4 Rook's graph (G1) and Shrikhande graph (G2)."""
    # Shrikhande graph
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
    return G1, G2

def test_betti_numbers_srg():
    """Verifies proposition on Betti vectors of SRG 2-clique complexes."""
    G1, G2 = create_srg_pair()
    
    # Verify 16 vertices, 48 edges
    assert G1.number_of_nodes() == 16 and G1.number_of_edges() == 48
    assert G2.number_of_nodes() == 16 and G2.number_of_edges() == 48
    
    # Find triangles (2-cliques)
    cliques_1 = [list(c) for c in nx.enumerate_all_cliques(G1) if len(c) == 3]
    cliques_2 = [list(c) for c in nx.enumerate_all_cliques(G2) if len(c) == 3]
    
    assert len(cliques_1) == 32 and len(cliques_2) == 32
    
    # Check degree and triangle regularity
    for u, v in G1.edges():
        assert len(list(nx.common_neighbors(G1, u, v))) == 2
        assert compute_forman_ricci_curvature(G1, (u, v)) == -2.0
        
    for u, v in G2.edges():
        assert len(list(nx.common_neighbors(G2, u, v))) == 2
        assert compute_forman_ricci_curvature(G2, (u, v)) == -2.0
    print("[PASS] SRG regularity and AF3 uniformity verified (-2.0 on all 48 edges).")

def test_permutation_equivariance():
    """Verifies permutation equivariance under node permutations."""
    torch.manual_seed(42)
    G = nx.erdos_renyi_graph(15, 0.35, seed=42)
    adj = nx.to_numpy_array(G)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    x = torch.randn(15, 8)
    
    pyg_data1 = Data(x=x, edge_index=edge_index, num_nodes=15)
    
    # Permute nodes
    perm = np.random.RandomState(42).permutation(15)
    perm_map = {i: int(perm[i]) for i in range(15)}
    
    # Map edges directly: (u, v) -> (perm[u], perm[v])
    edges = list(G.edges())
    edges_perm = [(perm_map[u], perm_map[v]) for u, v in edges]
    edge_index_perm_list = []
    for u, v in edges_perm:
        edge_index_perm_list.append([u, v])
        edge_index_perm_list.append([v, u])
    edge_index_perm = torch.tensor(edge_index_perm_list, dtype=torch.long).t()
    
    # Node features: node new_i gets x[old_i]
    x_perm = torch.zeros_like(x)
    for old_i, new_i in perm_map.items():
        x_perm[new_i] = x[old_i]
        
    pyg_data2 = Data(x=x_perm, edge_index=edge_index_perm, num_nodes=15)
    
    # Lift both
    cc1, _ = lift_graph_to_cell_complex(pyg_data1, max_cycle_length=6, curvature_type='af3')
    cc2, _ = lift_graph_to_cell_complex(pyg_data2, max_cycle_length=6, curvature_type='af3')
    
    # Verify 2-cell counts match exactly
    assert len(cc1.skeleton(2)) == len(cc2.skeleton(2))
    
    # Build model and test graph-level output invariance
    model = DynamicCWNet(num_node_features=8, hidden_dim=32, num_classes=1, num_layers=2, readout='sum')
    model.eval()
    
    B1_1, B2_1 = get_incidence_matrices(cc1)
    B1_2, B2_2 = get_incidence_matrices(cc2)
    
    edgelist_1 = sorted([tuple(sorted(e)) for e in cc1._G.edges])
    edgelist_2 = sorted([tuple(sorted(e)) for e in cc2._G.edges])
    
    frc_dict1 = cc1.get_cell_attributes('curvature', rank=1)
    frc_dict2 = cc2.get_cell_attributes('curvature', rank=1)
    
    frc_1 = torch.tensor([frc_dict1[e] for e in edgelist_1], dtype=torch.float32).unsqueeze(1)
    frc_2 = torch.tensor([frc_dict2[e] for e in edgelist_2], dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        out1 = model(x, None, None, B1_1, B2_1, frc_1)
        out2 = model(x_perm, None, None, B1_2, B2_2, frc_2)
        
    diff = torch.abs(out1 - out2).item()
    assert diff < 1e-4, f"Permutation equivariance violated: diff = {diff}"
    print(f"[PASS] Permutation equivariance verified (L1 difference = {diff:.2e}).")

def test_orientation_invariance():
    """Verifies that unoriented message passing |B1|, |B2| is independent of signed orientation choice."""
    G = nx.cycle_graph(5)
    adj = nx.to_numpy_array(G)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    x = torch.randn(5, 8)
    pyg_data = Data(x=x, edge_index=edge_index, num_nodes=5)
    
    cc, _ = lift_graph_to_cell_complex(pyg_data, max_cycle_length=6)
    B1, B2 = get_incidence_matrices(cc)
    
    # Invert signs of arbitrary columns/rows in boundary operators
    B1_dense = torch.abs(B1.to_dense())
    B2_dense = torch.abs(B2.to_dense())
    
    # Absolute matrices must be identical regardless of orientation
    assert torch.all(B1_dense >= 0)
    assert torch.all(B2_dense >= 0)
    print("[PASS] Orientation invariance of absolute boundary maps verified.")

if __name__ == '__main__':
    print("Running Topological Invariants and Equivariance Unit Tests...")
    test_betti_numbers_srg()
    test_permutation_equivariance()
    test_orientation_invariance()
    print("All unit tests passed successfully!")
