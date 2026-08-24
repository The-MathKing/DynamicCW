"""
Data Preprocessing module used for testing.
Handles the heavy lifting of mapping standard PyG geometric graphs into 
higher-dimensional TopoNetX Simplicial Complexes (up to 3-cells) and 
computing the discrete Forman-Ricci Curvature for the 1-cells.
"""
import torch
import networkx as nx
import toponetx as tnx
from torch_geometric.utils import to_networkx

def compute_forman_ricci_curvature(G, edge):
    """
    Computes a generalized discrete Forman-Ricci Curvature for an edge (1-simplex).
    Formula: FRC(e) = 4 - deg(v1) - deg(v2) + 3 * #(Triangles containing e)
    """
    v1, v2 = edge
    deg_v1 = G.degree(v1)
    deg_v2 = G.degree(v2)
    
    # Find number of triangles containing the edge
    # A triangle containing (v1, v2) means there is a node w connected to both v1 and v2.
    common_neighbors = list(nx.common_neighbors(G, v1, v2))
    num_triangles = len(common_neighbors)
    
    frc = 4 - deg_v1 - deg_v2 + 3 * num_triangles
    return frc

def lift_graph_to_simplicial_complex(pyg_data, max_cycle_length=6):
    """
    Lifts a PyTorch Geometric Data object (graph) to a TopoNetX CellComplex.
    Also computes Forman-Ricci Curvature for all 1-cells (edges) and adds it as a feature.
    """
    # Convert PyG Data to NetworkX Graph
    G = to_networkx(pyg_data, to_undirected=True)
    
    # Initialize a Cell Complex
    # We build a 2D CW complex where 2-cells are bounded chordless cycles
    CC = tnx.CellComplex(G)
    
    # Add 0-cells (nodes)
    for node in G.nodes():
        # Keep node features if they exist
        features = {}
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            features['x'] = pyg_data.x[node].numpy()
        CC.set_cell_attributes({node: features}, name='features', rank=0)
        
    # Add 1-cells (edges) and compute FRC
    for edge in G.edges():
        frc = compute_forman_ricci_curvature(G, edge)
        CC.set_cell_attributes({tuple(edge): frc}, name='frc', rank=1)
        
    # Add 2-cells (chordless cycles up to max_cycle_length)
    if max_cycle_length is not None and max_cycle_length > 2:
        # Find chordless cycles
        cycles = list(nx.chordless_cycles(G, length_bound=max_cycle_length))
        
        # Limit to max 5000 cycles to prevent memory blowup in dense graphs
        for cycle in cycles[:5000]:
            if len(cycle) > 2:
                CC.add_cell(cycle, rank=2)
                
    return CC, G

if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    # Test with a simple dataset
    print("Loading MUTAG dataset for testing...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    data = dataset[0]
    
    print(f"Graph Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    
    sc, G = lift_graph_to_simplicial_complex(data)
    
    print(f"Simplicial Complex shape:")
    print(f"  0-cells (nodes): {len(sc.skeleton(0))}")
    if sc.dim >= 1:
        print(f"  1-cells (edges): {len(sc.skeleton(1))}")
    if sc.dim >= 2:
        print(f"  2-cells (triangles): {len(sc.skeleton(2))}")
    
    # Check curvature of first few edges
    if sc.dim >= 1:
        print("Sample Edge Curvatures:")
        for i, edge in enumerate(list(sc.skeleton(1))[:5]):
            print(f"  Edge {edge}: FRC = {sc.get_cell_attributes('frc', rank=1)[tuple(edge)]}")
