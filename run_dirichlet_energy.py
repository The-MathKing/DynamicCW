"""
Rigorous Dirichlet Energy and Over-Smoothing Analysis on Cell Complexes.
Computes:
1. Normalized 0-Dirichlet Energy: E_0(H_V) = Tr(H_V^T Delta_0 H_V) / ||H_V||_F^2
   where Delta_0 = B1 B1^T (D - A), the signed graph Laplacian on vertices.
2. Normalized 1-Dirichlet Energy: E_1(H_E) = Tr(H_E^T Delta_1 H_E) / ||H_E||_F^2
   where Delta_1 is the unsigned edge-variation Laplacian built from lower- and
   upper-adjacency between edges (sharing a vertex / a 2-cell); orientation-free
   by construction, unlike B1^T B1 + B2 B2^T which depends on edge orientation.
Evaluates across depths L in [0..10] comparing:
- DynamicCW with Residuals & LayerNorm
- DynamicCW without Residuals (Unregularized)
- DynamicCW No-Gate baseline
- Standard 1-WL GIN baseline
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.datasets import TUDataset

from data_processing import lift_graph_to_cell_complex
from train import get_incidence_matrices
from model import DynamicCWNet

def compute_normalized_dirichlet_energies(H_V, H_E, B1, B2):
    """
    Computes normalized Dirichlet energies on 0-cells and 1-cells.
    Uses the SIGNED boundary matrices: B1 B1^T = D - A is the graph Laplacian
    (orientation-independent, since B1 B1^T is invariant to per-edge sign flips),
    unlike |B1||B1|^T = D + A, the signless Laplacian, which is maximized rather
    than minimized by smooth (near-constant) signals.
    """
    B1_d = B1.to_dense() if B1.is_sparse else B1
    Delta_0 = torch.matmul(B1_d, B1_d.t())

    # Node Dirichlet Energy
    norm_V = torch.norm(H_V, p='fro')**2 + 1e-8
    E_0 = torch.trace(torch.matmul(torch.matmul(H_V.t(), Delta_0), H_V)) / norm_V

    # Edge Dirichlet Energy: an orientation-free "edge variation" Laplacian built
    # from UNSIGNED lower/upper edge-adjacency (edges sharing a vertex / a 2-cell).
    # Unlike B1^T B1 + B2 B2^T (which involves the signed B1, B2 and is therefore
    # sensitive to the arbitrary per-edge orientation choice -- conjugating by a
    # sign-flip matrix changes the quadratic form), this adjacency is purely
    # combinatorial: A_lower[e,e'] = 1 iff e, e' share exactly one vertex, and
    # A_upper[e,e'] = (number of shared 2-cells), both built from |B1|, |B2|,
    # exactly mirroring Delta_0 = B1 B1^T = D - A on the node side.
    if H_E is not None and H_E.shape[0] > 0 and B1_d.shape[1] > 0:
        n_edges = B1_d.shape[1]
        absB1 = torch.abs(B1_d)
        lower_adj = torch.matmul(absB1.t(), absB1)
        lower_adj = lower_adj - torch.diag(torch.diag(lower_adj))
        D_lower = torch.diag(lower_adj.sum(dim=1))
        Delta_1 = D_lower - lower_adj

        if B2 is not None and B2.shape[1] > 0:
            B2_d = B2.to_dense() if B2.is_sparse else B2
            absB2 = torch.abs(B2_d)
            upper_adj = torch.matmul(absB2, absB2.t())
            upper_adj = upper_adj - torch.diag(torch.diag(upper_adj))
            D_upper = torch.diag(upper_adj.sum(dim=1))
            Delta_1 = Delta_1 + (D_upper - upper_adj)

        norm_E = torch.norm(H_E, p='fro')**2 + 1e-8
        E_1 = torch.trace(torch.matmul(torch.matmul(H_E.t(), Delta_1), H_E)) / norm_E
    else:
        E_1 = torch.tensor(0.0)

    return E_0.item(), E_1.item()

def run_dirichlet_analysis(num_layers=10, num_molecules=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading NCI1 dataset for Dirichlet Energy Analysis...")
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    
    molecules = []
    for data in dataset:
        cc, G = lift_graph_to_cell_complex(data, max_cycle_length=6, curvature_type='af3')
        if len(cc.skeleton(2)) > 0 and len(cc.skeleton(1)) > 5:
            B1, B2 = get_incidence_matrices(cc)
            edgelist = sorted([tuple(sorted(e)) for e in cc._G.edges])
            frc_dict = cc.get_cell_attributes('curvature', rank=1)
            frc = torch.tensor([frc_dict.get(e, 0.0) for e in edgelist], dtype=torch.float32).unsqueeze(1)
            x_0 = data.x.float() if data.x is not None else torch.ones((data.num_nodes, 1))
            molecules.append({'x_0': x_0, 'B1': B1, 'B2': B2, 'frc': frc})
            if len(molecules) >= num_molecules:
                break
                
    models = {
        'DynamicCW (With Residuals & Norm)': {'use_residuals': True, 'use_norm': True, 'gating': 'vector'},
        'DynamicCW (Unregularized, No Residuals)': {'use_residuals': False, 'use_norm': False, 'gating': 'vector'},
        'DynamicCW (No Gate, With Residuals)': {'use_residuals': True, 'use_norm': True, 'gating': 'none'},
    }
    
    results = {}
    hidden_dim = 32
    num_node_feats = molecules[0]['x_0'].shape[1]
    
    for name, cfg in models.items():
        print(f"Tracking energy decay across {num_layers} layers for: {name}")
        model = DynamicCWNet(
            num_node_features=num_node_feats,
            hidden_dim=hidden_dim,
            num_classes=1,
            num_layers=num_layers,
            gating=cfg['gating'],
            dynamic_faces=True,
            use_residuals=cfg['use_residuals'],
            use_norm=cfg['use_norm']
        )
        model.eval()
        
        all_mol_e0 = []
        all_mol_e1 = []
        
        for mol in molecules:
            x_0 = mol['x_0']
            B1 = mol['B1']
            B2 = mol['B2']
            frc = mol['frc']
            
            # Embed initial
            h_0 = model.node_embedding(x_0)
            h_1 = model.edge_embedding(torch.zeros((B1.shape[1], 8)))
            h_2 = model.face_embedding(torch.ones((B2.shape[1], 1))) if B2.shape[1] > 0 else None
            
            e0_traj = []
            e1_traj = []
            
            e0, e1 = compute_normalized_dirichlet_energies(h_0, h_1, B1, B2)
            e0_traj.append(e0)
            e1_traj.append(e1)
            
            for conv in model.convs:
                h_0, h_1, h_2 = conv(h_0, h_1, h_2, B1, B2, frc)
                e0, e1 = compute_normalized_dirichlet_energies(h_0, h_1, B1, B2)
                e0_traj.append(e0)
                e1_traj.append(e1)
                
            all_mol_e0.append(e0_traj)
            all_mol_e1.append(e1_traj)
            
        mean_e0 = np.mean(all_mol_e0, axis=0).tolist()
        std_e0 = np.std(all_mol_e0, axis=0).tolist()
        mean_e1 = np.mean(all_mol_e1, axis=0).tolist()
        std_e1 = np.std(all_mol_e1, axis=0).tolist()
        
        results[name] = {
            'mean_node_energy': mean_e0,
            'std_node_energy': std_e0,
            'mean_edge_energy': mean_e1,
            'std_edge_energy': std_e1
        }
        
    os.makedirs('results', exist_ok=True)
    with open('results/dirichlet_energy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Dirichlet energy trajectory results saved to results/dirichlet_energy_results.json")
    return results

if __name__ == '__main__':
    run_dirichlet_analysis(num_layers=10, num_molecules=20)
