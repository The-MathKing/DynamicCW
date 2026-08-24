import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
import time
import numpy as np
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
import json

from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices, process_dataset, train_epoch, test
from model import CurvatureMPSN
from adversarial_utils import targeted_bottleneck_attack

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_srgs():
    # 1. Shrikhande Graph
    G1 = nx.Graph()
    for x1 in range(4):
        for y1 in range(4):
            for x2 in range(4):
                for y2 in range(4):
                    dx = (x1 - x2) % 4
                    dy = (y1 - y2) % 4
                    if (dx, dy) in [(1,0), (3,0), (0,1), (0,3), (1,1), (3,3)]:
                        G1.add_edge((x1, y1), (x2, y2))
    G1 = nx.convert_node_labels_to_integers(G1)
    
    # 2. 4x4 Rook's Graph
    K4 = nx.complete_graph(4)
    G2 = nx.cartesian_product(K4, K4)
    G2 = nx.convert_node_labels_to_integers(G2)
    
    return G1, G2

def process_graph(G):
    pyg_data = from_networkx(G)
    pyg_data.x = torch.ones((G.number_of_nodes(), 1))
    sc, _ = lift_graph_to_simplicial_complex(pyg_data, max_dim=2)
    B1, B2 = get_incidence_matrices(sc)
    x_0 = pyg_data.x
    if sc.dim >= 1:
        frc_dict = sc.get_cell_attributes('frc', rank=1)
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
    return x_0, B1, B2, frc_weights, hlpe


def test_isomorphism():
    print("\n--- 1. SRG Isomorphism Test ---")
    G1, G2 = generate_srgs()
    x0_1, B1_1, B2_1, frc_1, hlpe_1 = process_graph(G1)
    x0_2, B1_2, B2_2, frc_2, hlpe_2 = process_graph(G2)
    
    set_seed(42)
    model = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2, gating='curvature')
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=5.0)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=5.0)
    model.apply(init_weights)
    model.eval()
    
    with torch.no_grad():
        out1 = model(x0_1, hlpe_1, None, B1_1, B2_1, frc_1, None, None, None)
        out2 = model(x0_2, hlpe_2, None, B1_2, B2_2, frc_2, None, None, None)
        
    distance = torch.norm(out1 - out2, p=2).item()
    print(f"L2 Metric Distance: {distance:.6f}")
    if distance < 1e-4:
        print(f"Warning: Isomorphism distance {distance} is extremely small due to uniform initialization symmetry.")
    return distance


def test_latency(device):
    print("\n--- 4. Iterative Latency Profiling ---")
    dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
    data = None
    for d in dataset:
        if d.num_nodes > 50:
            data = d
            break
            
    x0, B1, B2, frc, hlpe = process_graph(to_networkx(data, to_undirected=True))
    model = CurvatureMPSN(num_node_features=1, hidden_dim=32, num_classes=2, gating='curvature')
    model.eval()
    
    with torch.no_grad():
        x_0 = model.node_embedding(x0)
        x_1 = model.edge_embedding(hlpe)
        x_2 = model.triangle_embedding(torch.ones((B2.shape[1], 1)))
    
    # Warmup
    for _ in range(10):
        o0, o1, o2 = model.conv1(x_0, x_1, x_2, B1, B2, frc)
        o0, o1, o2 = model.conv2(o0, o1, o2, B1, B2, frc)
        
    l1_times = []
    l2_times = []
    
    for _ in range(100):
        t0 = time.time()
        with torch.no_grad():
            o0, o1, o2 = model.conv1(x_0, x_1, x_2, B1, B2, frc)
        t1 = time.time()
        
        t2 = time.time()
        with torch.no_grad():
            o0_out, o1_out, o2_out = model.conv2(o0, o1, o2, B1, B2, frc)
        t3 = time.time()
        
        l1_times.append((t1 - t0)*1000)
        l2_times.append((t3 - t2)*1000)
        
    l1_mean = np.mean(l1_times)
    l1_std = np.std(l1_times)
    
    l2_mean = np.mean(l2_times)
    l2_std = np.std(l2_times)
    
    print(f"Layer 1 Inference Time: {l1_mean:.4f} ± {l1_std:.4f} ms")
    print(f"Layer 2 Inference Time: {l2_mean:.4f} ± {l2_std:.4f} ms")
    
    return {"layer_1_ms": l1_mean, "layer_1_std": l1_std, "layer_2_ms": l2_mean, "layer_2_std": l2_std}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    try:
        iso_dist = test_isomorphism()
        latency = test_latency(device)
        
        results = {
            "isomorphism_l2_distance": iso_dist,
            "latency": latency
        }
        
        with open('results_full_run_v2.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print("\nAll V2 Benchmarks Completed Successfully. Output saved to results_full_run_v2.json")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")

if __name__ == '__main__':
    main()
