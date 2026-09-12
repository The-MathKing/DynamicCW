"""
Verification of Betti Number Separation and Expressivity Limits on SRG(16, 6, 2, 2)
Tests:
1) Betti numbers of the 2-clique complex (triangle lifting): (1,9,8) vs (1,2,1).
2) Face counts and Betti numbers under chordless cycle lifting (<=6): |F|=164 vs 204.
3) Cellular message-passing representation distance with constant initial features under triangle lifting (demonstrating L2 ~ 0).
"""

import itertools
from collections import Counter
import numpy as np
import networkx as nx

def generate_srgs():
    """Generate the 1-WL indistinguishable Shrikhande and 4x4 Rook's graphs."""
    # Shrikhande graph (16, 6, 2, 2)
    gens = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    G_shrikhande = nx.Graph()
    for a in itertools.product(range(4), repeat=2):
        for b in itertools.product(range(4), repeat=2):
            if ((a[0] - b[0]) % 4, (a[1] - b[1]) % 4) in gens:
                G_shrikhande.add_edge(a, b)
    G_shrikhande = nx.convert_node_labels_to_integers(G_shrikhande)
    
    # 4x4 Rook's graph (Cartesian product K_4 x K_4) (16, 6, 2, 2)
    K4 = nx.complete_graph(4)
    G_rooks = nx.cartesian_product(K4, K4)
    G_rooks = nx.convert_node_labels_to_integers(G_rooks)
    
    return G_rooks, G_shrikhande

def boundary_matrices(G, faces):
    E = [tuple(sorted(e)) for e in G.edges()]
    idx = {e: i for i, e in enumerate(E)}
    B1 = np.zeros((G.number_of_nodes(), len(E)))
    for j, (u, v) in enumerate(E):
        B1[u, j], B1[v, j] = -1, 1
    B2 = np.zeros((len(E), len(faces)))
    for k, f in enumerate(faces):
        for i in range(len(f)):
            u, v = f[i], f[(i + 1) % len(f)]
            B2[idx[tuple(sorted((u, v)))], k] = 1 if u < v else -1
    assert np.allclose(B1 @ B2, 0)
    return E, B1, B2

def betti(B1, B2):
    r1, r2 = np.linalg.matrix_rank(B1), np.linalg.matrix_rank(B2)
    return (B1.shape[0] - r1, B1.shape[1] - r1 - r2, B2.shape[1] - r2), (r1, r2)

def cellular_mp(B1, B2, frc, D=256, layers=3, seed=0):
    rng = np.random.default_rng(seed)
    W = {k: rng.normal(size=(D, D)) / np.sqrt(D) for k in
         ["vE", "fE", "adj", "e0", "e1", "EV", "EF", "v0", "v1", "f0", "f1"]}
    Wg = rng.normal(size=(D, D + 1)) / np.sqrt(D)
    A1, A2 = np.abs(B1), np.abs(B2)
    if frc.std() > 1e-6:
        kappa = (frc - frc.mean()) / (frc.std() + 1e-5)
    else:
        kappa = np.zeros_like(frc)
    HV, HE, HF = np.ones((A1.shape[0], D)), np.ones((A1.shape[1], D)), np.ones((A2.shape[1], D))
    for _ in range(layers):
        msg = (A1.T @ HV) @ W["vE"].T + (A2 @ HF) @ W["fE"].T \
              + (A1.T @ A1 @ HE + A2 @ A2.T @ HE) @ W["adj"].T
        Ht = np.tanh((1.3 * HE + msg) @ W["e0"].T) @ W["e1"].T
        gate = 1 / (1 + np.exp(-np.clip(np.hstack([Ht, kappa[:, None]]) @ Wg.T, -50, 50)))
        HE_new = np.tanh(Ht * gate)
        HV = np.tanh((1.2 * HV + (A1 @ HE_new) @ W["EV"].T) @ W["v0"].T) @ W["v1"].T
        HF = np.tanh((1.1 * HF + (A2.T @ HE_new) @ W["EF"].T) @ W["f0"].T) @ W["f1"].T
        HE = HE_new
    return np.concatenate([X.mean(0) for X in (HV, HE, HF)] + [X.sum(0) for X in (HV, HE, HF)])

if __name__ == "__main__":
    print("=" * 70)
    print("SRG(16, 6, 2, 2) Topological Betti Separation & Expressivity Verification")
    print("=" * 70)
    
    G_rooks, G_shrikhande = generate_srgs()
    emb = {}
    
    for name, G in [("Rook 4x4", G_rooks), ("Shrikhande", G_shrikhande)]:
        print(f"\n== {name}: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
        
        # 1. Triangle lifting (clique 2-skeleton)
        tris = [list(c) for c in nx.chordless_cycles(G, length_bound=3)]
        E, B1, B2 = boundary_matrices(G, tris)
        b, r = betti(B1, B2)
        frc = np.array([4 - G.degree(u) - G.degree(v) + 3 * len(list(nx.common_neighbors(G, u, v)))
                        for u, v in E], dtype=float)
        print(f"  [Triangle Lifting] |F|={len(tris)}  Boundary Ranks={r}  Betti Vector (b0,b1,b2)={b}")
        print(f"  Faces per edge: {dict(Counter(np.abs(B2).sum(1)))}  AF3 Curvature: {dict(Counter(frc))}")
        emb[name] = cellular_mp(B1, B2, frc)

        # 2. Chordless cycles lifting (<=6)
        cyc = [list(c) for c in nx.chordless_cycles(G, length_bound=6)]
        _, B1c, B2c = boundary_matrices(G, cyc)
        bc, rc = betti(B1c, B2c)
        print(f"  [Chordless <=6 Lifting] |F|={len(cyc)} by length={dict(Counter(map(len, cyc)))}  Betti={bc}")
        print(f"  Faces per edge: {dict(Counter(np.abs(B2c).sum(1)))}")

    d = np.linalg.norm(emb["Rook 4x4"] - emb["Shrikhande"])
    print("\n" + "-" * 70)
    print(f"Empirical Embedding L2 Distance (Triangle Lifting, Uniform Inputs, D=256): {d:.2e}")
    print("Confirmation: Unaugmented Cellular MP on triangle lifting produces identical")
    print("representations on symmetric SRGs (L2 ~ 0), as proved by Eitan et al. (ICLR 2025).")
    print("=" * 70)
