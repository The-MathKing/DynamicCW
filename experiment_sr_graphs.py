import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from data_processing import lift_graph_to_simplicial_complex
from model import CurvatureMPSN
from model_baselines import BaselineGCN

def get_incidence_matrices(sc):
    if sc.dim >= 1:
        B1_scipy = sc.incidence_matrix(rank=1, signed=True)
        coo = B1_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        B1 = torch.sparse_coo_tensor(indices, values, size=coo.shape)
    else:
        B1 = torch.zeros((len(sc.skeleton(0)), 0))
        
    if sc.dim >= 2:
        B2_scipy = sc.incidence_matrix(rank=2, signed=True)
        coo = B2_scipy.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        B2 = torch.sparse_coo_tensor(indices, values, size=coo.shape)
    else:
        B2 = torch.zeros((len(sc.skeleton(1)), 0))
        
    return B1, B2

def generate_sr_dataset():
    """
    Generates a dataset of Strongly Regular Graphs.
    Class 0: Shrikhande Graph (16, 6, 2, 2)
    Class 1: Rook's Graph (4x4, Cartesian product of K4 x K4, also 16, 6, 2, 2)
    """
    dataset = []
    
    # 1. Shrikhande Graph
    # Mathematically defined as a Cayley graph on Z4 x Z4 with specific generators
    G_shrikhande = nx.Graph()
    vertices = [(i, j) for i in range(4) for j in range(4)]
    G_shrikhande.add_nodes_from(vertices)
    generators = [(1,0), (3,0), (0,1), (0,3), (1,1), (3,3)]
    for u in vertices:
        for g in generators:
            v = ((u[0] + g[0]) % 4, (u[1] + g[1]) % 4)
            G_shrikhande.add_edge(u, v)
    G_shrikhande = nx.convert_node_labels_to_integers(G_shrikhande)
    pyg_shrikhande = from_networkx(G_shrikhande)
    # 1-WL is colorblind to structural differences if degrees are identical,
    # so we initialize identical features (all ones) for all nodes.
    pyg_shrikhande.x = torch.ones((pyg_shrikhande.num_nodes, 1), dtype=torch.float)
    pyg_shrikhande.y = torch.tensor([0], dtype=torch.long)
    dataset.append(pyg_shrikhande)
    
    # 2. Rook's Graph (K4 x K4)
    K4 = nx.complete_graph(4)
    G_rooks = nx.cartesian_product(K4, K4)
    G_rooks = nx.convert_node_labels_to_integers(G_rooks)
    pyg_rooks = from_networkx(G_rooks)
    pyg_rooks.x = torch.ones((pyg_rooks.num_nodes, 1), dtype=torch.float)
    pyg_rooks.y = torch.tensor([1], dtype=torch.long)
    dataset.append(pyg_rooks)
    
    return dataset

def run_experiment():
    print("=== EXPERIMENT A: SYNTHETIC EXPRESSIVITY TEST (1-WL BREAK) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = generate_sr_dataset()
    
    # Preprocess dataset for MPSN
    processed_mpsn = []
    for data in dataset:
        sc, _ = lift_graph_to_simplicial_complex(data)
        B1, B2 = get_incidence_matrices(sc)
        x_0 = data.x
        
        if sc.dim >= 1:
            frc_dict = sc.get_simplex_attributes('frc')
            frc_list = [frc_dict[tuple(edge)] for edge in sc.skeleton(1)]
            frc_weights = torch.tensor(frc_list, dtype=torch.float32).unsqueeze(1)
        else:
            frc_weights = torch.empty((0, 1))
            
        processed_mpsn.append({
            'x_0': x_0, 'B1': B1, 'B2': B2, 'frc': frc_weights, 'y': data.y
        })
        
    print("Datasets Generated and Lifted to Simplicial Complexes.")
    
    # We want to train models to classify whether the graph is Shrikhande (0) or Rook's (1).
    # Since there are only 2 graphs, we will heavily overfit on them to see if the model 
    # CAN physically distinguish them (achieve 100% train accuracy).
    
    def train_and_eval(model, model_type="gcn"):
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            total_loss = 0
            
            for i in range(2):
                if model_type == "gcn":
                    data = dataset[i].to(device)
                    out = model(data.x, data.edge_index)
                    y = data.y
                else:
                    data = processed_mpsn[i]
                    out = model(data['x_0'].to(device), None, None, 
                                data['B1'].to(device), data['B2'].to(device), 
                                data['frc'].to(device), None, None, None)
                    y = data['y'].to(device)
                    
                loss = criterion(out, y)
                loss.backward()
                total_loss += loss.item()
                
            optimizer.step()
            
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(2):
                if model_type == "gcn":
                    data = dataset[i].to(device)
                    out = model(data.x, data.edge_index)
                else:
                    data = processed_mpsn[i]
                    out = model(data['x_0'].to(device), None, None, 
                                data['B1'].to(device), data['B2'].to(device), 
                                data['frc'].to(device), None, None, None)
                pred = out.argmax(dim=1)
                if pred == dataset[i].y.to(device):
                    correct += 1
                    
        return correct / 2.0

    # 1. Baseline 1-WL GCN
    print("\nTraining Standard 1-WL GCN...")
    gcn = BaselineGCN(num_node_features=1, hidden_dim=32, num_classes=2).to(device)
    gcn_acc = train_and_eval(gcn, model_type="gcn")
    print(f"GCN Accuracy: {gcn_acc*100:.2f}% (Expected: 50.00% because 1-WL fails to distinguish SR graphs)")
    
    # 2. Curvature-Weighted MPSN
    print("\nTraining Curvature-Weighted MPSN...")
    mpsn = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2).to(device)
    mpsn_acc = train_and_eval(mpsn, model_type="mpsn")
    print(f"MPSN Accuracy: {mpsn_acc*100:.2f}% (Expected: 100.00% because curvature weighting breaks 1-WL limit)")

if __name__ == "__main__":
    run_experiment()
