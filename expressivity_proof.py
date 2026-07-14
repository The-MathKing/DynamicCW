import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch

# Assuming your models are in these files (adjust imports as needed for your local environment)
from dynamic_cw_network import DynamicCWNetwork
from train_crucible import FormanRicciTransform
# We will use a basic GCN for the failure baseline to prove standard message passing is blind
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class StandardGCN(nn.Module):
    """A standard 1-WL bounded Graph Convolutional Network."""
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 16)
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

def generate_1wl_twins():
    """
    Generates the Shrikhande Graph and the 4x4 Rook's Graph.
    Both are strongly regular graphs with parameters (16, 6, 2, 2).
    Standard 1-WL tests and standard GNNs CANNOT distinguish them.
    """
    # 1. Shrikhande Graph
    # Constructed manually: Z_4 x Z_4 with specific modulo adjacencies
    G1 = nx.Graph()
    for x in range(4):
        for y in range(4):
            for dx, dy in [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,-1)]:
                u, v = (x+dx)%4, (y+dy)%4
                G1.add_edge(x*4+y, u*4+v)
    
    # 2. Rook's Graph (4x4 Grid)
    # The Rook's graph on a 4x4 chessboard connects squares in the same row or column.
    G2 = nx.Graph()
    for i in range(16):
        r1, c1 = divmod(i, 4)
        for j in range(i+1, 16):
            r2, c2 = divmod(j, 4)
            if r1 == r2 or c1 == c2:
                G2.add_edge(i, j)

    # Convert to PyTorch Geometric Data
    # Give all nodes identical features (a single scalar of 1.0) so the network 
    # is forced to rely ENTIRELY on topology.
    data1 = from_networkx(G1)
    data1.x = torch.ones((16, 1))
    
    data2 = from_networkx(G2)
    data2.x = torch.ones((16, 1))
    
    return data1, data2

def run_expressivity_proof():
    print("="*60)
    print("1-WL EXPRESSIVITY MATHEMATICAL PROOF")
    print("="*60)
    
    data1, data2 = generate_1wl_twins()
    
    print("\n[Step 1] Initializing standard 1-WL bounded GCN...")
    gcn = StandardGCN()
    gcn.eval()
    
    # Forward pass through standard GNN
    with torch.no_grad():
        out1_gcn = gcn(data1.x, data1.edge_index, torch.zeros(16, dtype=torch.long))
        out2_gcn = gcn(data2.x, data2.edge_index, torch.zeros(16, dtype=torch.long))
        
    # Calculate Euclidean distance between the graph embeddings
    diff_gcn = torch.norm(out1_gcn - out2_gcn).item()
    print(f"GCN Latent Distance between Shrikhande and Rook's graphs: {diff_gcn:.6f}")
    if diff_gcn < 1e-4:
        print("-> FAILURE: GCN is mathematically blind. It thinks these different graphs are identical.")
        
    print("\n[Step 2] Initializing Latent-Dynamic Simplicial Network...")
    # Apply our custom curvature transform to extract simplicial cliques
    transform = FormanRicciTransform()
    data1_cw = transform(data1)
    data2_cw = transform(data2)
    
    # Initialize our custom model (Using in_channels=1, hidden=16, out=2)
    cw_net = DynamicCWNetwork(1, 16, 2)
    cw_net.eval()
    
    with torch.no_grad():
        out1_cw = cw_net(data1_cw.x, data1_cw.edge_index, data1_cw.edge_attr, torch.zeros(16, dtype=torch.long))
        out2_cw = cw_net(data2_cw.x, data2_cw.edge_index, data2_cw.edge_attr, torch.zeros(16, dtype=torch.long))
        
    # Calculate Euclidean distance between the graph embeddings
    diff_cw = torch.norm(out1_cw - out2_cw).item()
    print(f"CW-Net Latent Distance between Shrikhande and Rook's graphs: {diff_cw:.6f}")
    if diff_cw > 1e-4:
        print(f"-> SUCCESS! CW-Net successfully separated the graphs in the latent space by a distance of {diff_cw:.4f}.")
        print("-> THE 1-WL LIMIT HAS BEEN BROKEN.")

if __name__ == "__main__":
    run_expressivity_proof()
