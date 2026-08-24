import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices
from model import CurvatureMPSN
from model_baselines import BaselineGCN
import random

device = torch.device('cpu')

def find_terminal_edge(G):
    for u, d in G.degree():
        if d == 1:
            # return the edge connected to this node
            neighbors = list(G.neighbors(u))
            return (u, neighbors[0])
    return None

def find_cycle_edge(G):
    cycles = nx.cycle_basis(G)
    if not cycles:
        return None
    # find a chordless cycle
    for c in cycles:
        sub = G.subgraph(c)
        if sub.number_of_edges() == len(c):
            return (c[0], c[1])
    return None

def process(data):
    sc, _ = lift_graph_to_simplicial_complex(data)
    B1, B2 = get_incidence_matrices(sc)
    
    if sc.dim >= 1:
        frc_dict = sc.get_cell_attributes('frc', rank=1)
        frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
        frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
    else:
        frc_weights = torch.empty((0, 1))
        
    return {
        'x_0': data.x,
        'edge_index': data.edge_index,
        'B1': B1,
        'B2': B2,
        'frc': frc_weights,
        'batch_0': torch.zeros(data.x.shape[0], dtype=torch.long),
        'batch_1': torch.zeros(max(1, B1.shape[1]), dtype=torch.long),
        'batch_2': torch.zeros(max(1, B2.shape[1]), dtype=torch.long)
    }

def main():
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    
    valid_graphs = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        G = to_networkx(data, to_undirected=True)
        G = nx.Graph(G)
        
        terminal_edge = find_terminal_edge(G)
        cycle_edge = find_cycle_edge(G)
        
        if terminal_edge and cycle_edge:
            valid_graphs.append((data, G, terminal_edge, cycle_edge))
            if len(valid_graphs) >= 100:
                break
                
    print(f"Found {len(valid_graphs)} valid molecules with both terminal and cycle edges.")
    
    torch.manual_seed(42)
    random.seed(42)
    
    gcn_ratios = []
    mpsn_ratios = []
    
    mpsn = CurvatureMPSN(num_node_features=dataset.num_node_features, hidden_dim=32, num_classes=2, gating='vector').to(device)
    gcn = BaselineGCN(num_node_features=dataset.num_node_features, hidden_dim=32, num_classes=2).to(device)
    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
    mpsn.apply(init_weights)
    gcn.apply(init_weights)
    
    mpsn.eval()
    gcn.eval()
    
    for data, G, term_edge, cyc_edge in valid_graphs:
        # Create versions
        G_base = G.copy()
        
        G_control = G.copy()
        G_control.remove_edge(*term_edge)
        
        G_topo = G.copy()
        G_topo.remove_edge(*cyc_edge)
        
        def to_data(graph, orig_data):
            d = from_networkx(graph)
            d.x = orig_data.x
            return d
            
        data_base = to_data(G_base, data)
        data_control = to_data(G_control, data)
        data_topo = to_data(G_topo, data)
        
        base_input = process(data_base)
        control_input = process(data_control)
        topo_input = process(data_topo)
        
        with torch.no_grad():
            gcn_base = gcn(base_input['x_0'], base_input['edge_index'])
            gcn_control = gcn(control_input['x_0'], control_input['edge_index'])
            gcn_topo = gcn(topo_input['x_0'], topo_input['edge_index'])
            
            mpsn_base = mpsn(base_input['x_0'], None, None, base_input['B1'], base_input['B2'], base_input['frc'], base_input['batch_0'], base_input['batch_1'], base_input['batch_2'])
            mpsn_control = mpsn(control_input['x_0'], None, None, control_input['B1'], control_input['B2'], control_input['frc'], control_input['batch_0'], control_input['batch_1'], control_input['batch_2'])
            mpsn_topo = mpsn(topo_input['x_0'], None, None, topo_input['B1'], topo_input['B2'], topo_input['frc'], topo_input['batch_0'], topo_input['batch_1'], topo_input['batch_2'])
            
        gcn_dist_control = torch.norm(gcn_base - gcn_control).item()
        gcn_dist_topo = torch.norm(gcn_base - gcn_topo).item()
        gcn_ratio = gcn_dist_topo / (gcn_dist_control + 1e-9)
        gcn_ratios.append(gcn_ratio)
        
        mpsn_dist_control = torch.norm(mpsn_base - mpsn_control).item()
        mpsn_dist_topo = torch.norm(mpsn_base - mpsn_topo).item()
        mpsn_ratio = mpsn_dist_topo / (mpsn_dist_control + 1e-9)
        mpsn_ratios.append(mpsn_ratio)
        
    gcn_mean = np.mean(gcn_ratios)
    gcn_std = np.std(gcn_ratios)
    
    mpsn_mean = np.mean(mpsn_ratios)
    mpsn_std = np.std(mpsn_ratios)
    
    print(f"GCN Sensitivity Ratio (Topo/Control): {gcn_mean:.2f}x ± {gcn_std:.2f}")
    print(f"MPSN Sensitivity Ratio (Topo/Control): {mpsn_mean:.2f}x ± {mpsn_std:.2f}")

if __name__ == '__main__':
    main()
