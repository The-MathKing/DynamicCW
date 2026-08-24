import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from data_processing import lift_graph_to_simplicial_complex
from train import get_incidence_matrices

def generate_srgs():
    # Shrikhande
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
    
    # Rook's
    K4 = nx.complete_graph(4)
    G2 = nx.cartesian_product(K4, K4)
    G2 = nx.convert_node_labels_to_integers(G2)
    return G1, G2

G1, G2 = generate_srgs()
sc1, _ = lift_graph_to_simplicial_complex(from_networkx(G1), max_dim=2)
sc2, _ = lift_graph_to_simplicial_complex(from_networkx(G2), max_dim=2)

print(f"Shrikhande: V={len(sc1.skeleton(0))}, E={len(sc1.skeleton(1))}, F={len(sc1.skeleton(2))}")
print(f"Rook's: V={len(sc2.skeleton(0))}, E={len(sc2.skeleton(1))}, F={len(sc2.skeleton(2))}")
