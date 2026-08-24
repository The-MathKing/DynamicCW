import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices
from model import CurvatureWeightedSimplicialConv
from torch_geometric.utils import to_networkx
import random
import warnings
warnings.filterwarnings('ignore')

class DeepCurvatureMPSN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers=10, gating='vector'):
        super(DeepCurvatureMPSN, self).__init__()
        self.node_embedding = nn.Linear(num_node_features, hidden_dim)
        self.edge_embedding = nn.Linear(8, hidden_dim)
        self.triangle_embedding = nn.Linear(1, hidden_dim)
        
        self.layers = nn.ModuleList([
            CurvatureWeightedSimplicialConv(hidden_dim, hidden_dim, gating=gating)
            for _ in range(num_layers)
        ])
        
    def forward_with_energy(self, x_0, x_1, x_2, incidence_1, incidence_2, frc):
        energies = []
        
        # Initial projection
        h_0 = self.node_embedding(x_0)
        h_1 = self.edge_embedding(x_1) if x_1 is not None and x_1.shape[0] > 0 else None
        h_2 = self.triangle_embedding(x_2) if x_2 is not None and x_2.shape[0] > 0 else None
        
        def calc_dirichlet_energy(nodes, edge_index):
            if edge_index is None or edge_index.shape[1] == 0:
                return 0.0
            src = nodes[edge_index[0]]
            dst = nodes[edge_index[1]]
            return torch.mean(torch.sum((src - dst)**2, dim=1)).item()
            
        edge_index = None
        if incidence_1 is not None and incidence_1.shape[1] > 0:
            B1_d = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1
            # Extract edge index from B1
            edge_list = []
            for j in range(B1_d.shape[1]):
                nodes = torch.where(B1_d[:, j] != 0)[0]
                if len(nodes) == 2:
                    edge_list.append([nodes[0].item(), nodes[1].item()])
                    edge_list.append([nodes[1].item(), nodes[0].item()])
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list).t()
                
        energies.append(calc_dirichlet_energy(h_0, edge_index))
        
        for layer in self.layers:
            h_0, h_1, h_2 = layer(h_0, h_1, h_2, incidence_1, incidence_2, frc)
            energies.append(calc_dirichlet_energy(h_0, edge_index))
            
        return energies

def test_dirichlet_energy():
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    torch.manual_seed(42)
    random.seed(42)
    
    # Find a molecule with edges and faces to properly test
    valid_data = None
    for data in dataset:
        sc, _ = lift_graph_to_simplicial_complex(data, max_dim=2)
        if sc.dim >= 2 and sc.skeleton(1):
            valid_data = data
            break
            
    if valid_data is None:
        valid_data = dataset[0]
        sc, _ = lift_graph_to_simplicial_complex(valid_data, max_dim=2)
        
    B1, B2 = get_incidence_matrices(sc)
    
    if sc.dim >= 1:
        frc_dict = sc.get_simplex_attributes('frc')
        frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
        frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
        
        B1_d = B1.to_dense() if B1.is_sparse else B1
        B2_d = B2.to_dense() if B2.is_sparse else B2
        L1 = torch.matmul(B1_d.t(), B1_d) + torch.matmul(B2_d, B2_d.t())
        eigvals, eigvecs = torch.linalg.eigh(L1)
        k = 8
        if eigvecs.shape[1] >= k:
            hlpe = eigvecs[:, :k]
        else:
            hlpe = torch.nn.functional.pad(eigvecs, (0, k - eigvecs.shape[1]))
    else:
        frc_weights = torch.empty((0, 1))
        hlpe = torch.empty((0, 8))
        
    x_0 = valid_data.x if valid_data.x is not None else torch.ones((valid_data.num_nodes, 1))
    x_2 = torch.ones((B2.shape[1], 1)) if B2.shape[1] > 0 else torch.empty((0, 1))
    
    model = DeepCurvatureMPSN(num_node_features=x_0.shape[1], hidden_dim=32, num_layers=10)
    model.eval()
    
    with torch.no_grad():
        energies = model.forward_with_energy(x_0, hlpe, x_2, B1, B2, frc_weights)
        
    print("Dirichlet Energy Trajectory:")
    for i, e in enumerate(energies):
        print(f"T_{i} = {e:.4f}")

if __name__ == '__main__':
    test_dirichlet_energy()
