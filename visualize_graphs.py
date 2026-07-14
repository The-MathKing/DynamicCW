import networkx as nx
import matplotlib.pyplot as plt
import os

def create_shrikhande():
    G_shrikhande = nx.Graph()
    vertices = [(i, j) for i in range(4) for j in range(4)]
    G_shrikhande.add_nodes_from(vertices)
    generators = [(1,0), (3,0), (0,1), (0,3), (1,1), (3,3)]
    for u in vertices:
        for g in generators:
            v = ((u[0] + g[0]) % 4, (u[1] + g[1]) % 4)
            G_shrikhande.add_edge(u, v)
    return G_shrikhande

def create_rooks():
    K4 = nx.complete_graph(4)
    G_rooks = nx.cartesian_product(K4, K4)
    return G_rooks

def visualize():
    G1 = create_shrikhande()
    G2 = create_rooks()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Draw Shrikhande
    pos1 = nx.spring_layout(G1, seed=42)
    nx.draw(G1, pos1, ax=axes[0], with_labels=False, node_color='skyblue', 
            node_size=500, edge_color='gray', linewidths=1, font_size=10)
    axes[0].set_title("Shrikhande Graph\n(16 nodes, degree 6)", fontsize=16)
    
    # Draw Rook's
    pos2 = nx.spring_layout(G2, seed=42)
    nx.draw(G2, pos2, ax=axes[1], with_labels=False, node_color='lightcoral', 
            node_size=500, edge_color='gray', linewidths=1, font_size=10)
    axes[1].set_title("Rook's Graph\n(16 nodes, degree 6)", fontsize=16)
    
    plt.tight_layout()
    
    out_path = '/Users/aryanpadarthi/.gemini/antigravity-ide/brain/bea06fd0-242f-4fa0-b620-0afb73ded386/sr_graphs_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize()
