"""
Data Preprocessing and Topological Lifting Module.
Handles lifting discrete graphs into 2D regular cell complexes (chordless cycle lifting / clique lifting)
and computing discrete Ricci curvature variants (AF3, Degree-Only, Cycle-Aware Forman, Shuffled, and Random).
Designed for strict permutation equivariance and reproducible benchmarks.
"""
import torch
import numpy as np
import networkx as nx
import toponetx as tnx
from torch_geometric.utils import to_networkx

def compute_forman_ricci_curvature(G, edge):
    """
    Computes standard Augmented Forman-Ricci Curvature (AF3) for an edge (1-cell).
    Formula: AF_3(e) = 4 - deg(u) - deg(v) + 3 * #(triangular 2-cells containing e)
    """
    u, v = edge
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    common_neighbors = list(nx.common_neighbors(G, u, v))
    num_triangles = len(common_neighbors)
    return 4.0 - deg_u - deg_v + 3.0 * num_triangles

def compute_degree_only_curvature(G, edge):
    """
    Computes degree-only curvature baseline.
    Formula: kappa_deg(e) = 4 - deg(u) - deg(v)
    Isolates whether the gate benefits from combinatorial topology (triangles/faces)
    or solely from endpoint degree statistics.
    """
    u, v = edge
    return 4.0 - G.degree(u) - G.degree(v)

def compute_cycle_aware_forman(G, edge, face_lengths_for_edge):
    """
    Computes exact Cycle-Aware Augmented Forman Curvature.
    Formula: AF_cycle(e) = 4 - deg(u) - deg(v) + sum_{f in faces(e)} (6 - |f|)
    Under regular cell complexes where 2-cells are chordless cycles of length |f|.
    """
    u, v = edge
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    face_contrib = sum(6.0 - length for length in face_lengths_for_edge)
    return 4.0 - deg_u - deg_v + face_contrib

def find_chordless_cycles_canonical(G, max_cycle_length=6):
    """
    Extracts chordless cycles in G up to length max_cycle_length in a canonical,
    deterministic, and permutation-equivariant manner.
    Cycles are canonically represented by their lexicographically minimum rotation.
    """
    if max_cycle_length <= 2:
        return []
    
    # Extract chordless cycles
    raw_cycles = list(nx.chordless_cycles(G, length_bound=max_cycle_length))
    
    canonical_cycles = []
    seen = set()
    for cycle in raw_cycles:
        if len(cycle) < 3 or len(cycle) > max_cycle_length:
            continue
        # Canonical representation of undirected cycle:
        # Find minimum element, rotate, and choose direction with smaller second element
        min_idx = cycle.index(min(cycle))
        rot1 = cycle[min_idx:] + cycle[:min_idx]
        rot2 = [rot1[0]] + rot1[1:][::-1]
        canon = tuple(rot1) if rot1 <= rot2 else tuple(rot2)
        if canon not in seen:
            seen.add(canon)
            canonical_cycles.append(list(canon))
            
    # Sort deterministically by (length, canonical node tuple) for stable ordering
    canonical_cycles.sort(key=lambda c: (len(c), c))
    return canonical_cycles

def lift_graph_to_cell_complex(pyg_data, max_cycle_length=6, curvature_type='af3', seed=None):
    """
    Lifts a PyTorch Geometric Data object into a 2D regular cell complex (TopoNetX CellComplex).
    Computes incidence matrices and edge curvature features in an equivariant, reproducible manner.
    
    Parameters:
      pyg_data: PyG Data object (with x, edge_index, y, etc.)
      max_cycle_length: max length of chordless cycles to include as 2-cells (default 6)
      curvature_type: 'af3', 'degree_only', 'cycle_aware', 'shuffled', 'random', or 'none'
      seed: random seed for stochastic baselines
    """
    G_raw = to_networkx(pyg_data, to_undirected=True)
    num_nodes = pyg_data.num_nodes if hasattr(pyg_data, 'num_nodes') and pyg_data.num_nodes is not None else len(G_raw.nodes)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(G_raw.edges())
                
    CC = tnx.CellComplex(G)
    
    # 0-cells (nodes)
    for node in sorted(G.nodes()):
        features = {}
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            features['x'] = pyg_data.x[node].cpu().numpy()
        CC.set_cell_attributes({node: features}, name='features', rank=0)
        
    # Extract canonical chordless cycles as 2-cells
    canonical_cycles = find_chordless_cycles_canonical(G, max_cycle_length=max_cycle_length)
    for cycle in canonical_cycles:
        CC.add_cell(cycle, rank=2)
        
    # Pre-map edges to incident face lengths for cycle-aware curvature
    edge_to_face_lengths = {}
    for edge in G.edges():
        u, v = sorted(edge)
        edge_to_face_lengths[(u, v)] = []
        
    for cycle in canonical_cycles:
        k = len(cycle)
        for i in range(k):
            u, v = sorted((cycle[i], cycle[(i + 1) % k]))
            if (u, v) in edge_to_face_lengths:
                edge_to_face_lengths[(u, v)].append(k)
                
    # 1-cells (edges) curvature computation
    curvatures = []
    edges_ordered = sorted(G.edges(), key=lambda e: (min(e), max(e)))
    for edge in edges_ordered:
        u, v = sorted(edge)
        if curvature_type == 'af3':
            c_val = compute_forman_ricci_curvature(G, (u, v))
        elif curvature_type == 'degree_only':
            c_val = compute_degree_only_curvature(G, (u, v))
        elif curvature_type == 'cycle_aware':
            c_val = compute_cycle_aware_forman(G, (u, v), edge_to_face_lengths.get((u, v), []))
        elif curvature_type == 'random':
            rng = np.random.default_rng(seed)
            c_val = float(rng.standard_normal())
        elif curvature_type == 'none':
            c_val = 0.0
        else: # default to af3
            c_val = compute_forman_ricci_curvature(G, (u, v))
            
        curvatures.append(c_val)
        CC.set_cell_attributes({(u, v): c_val}, name='curvature', rank=1)
        
    if curvature_type == 'shuffled':
        rng = np.random.default_rng(seed)
        shuffled_curv = rng.permutation(curvatures)
        for i, edge in enumerate(edges_ordered):
            u, v = sorted(edge)
            CC.set_cell_attributes({(u, v): float(shuffled_curv[i])}, name='curvature', rank=1)
            
    return CC, G

def lift_graph_to_simplicial_complex(pyg_data, max_cycle_length=6, max_dim=None, **kwargs):
    """
    Backward-compatible wrapper for previous scripts.
    """
    curvature_type = kwargs.get('curvature_type', 'af3')
    return lift_graph_to_cell_complex(pyg_data, max_cycle_length=max_cycle_length, curvature_type=curvature_type)
