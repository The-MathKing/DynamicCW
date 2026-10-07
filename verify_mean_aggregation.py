"""
Scope check for Theorem 1: the expressivity ceiling assumes injective (sum-like)
aggregation, i.e. message passing bounded by 1-CWL. With mean (degree-normalized)
aggregation and a mean readout, degree is no longer recoverable, and a Forman gate
CAN add distinguishing information -- but only if the curvature is fed in raw;
per-graph standardization maps any uniform-curvature graph to kappa_tilde = 0.

Pair: cycle C_8 (2-regular, AF3 = 0 on every edge) vs. cube Q_3 (3-regular,
AF3 = -2 on every edge). Both have 8 vertices and no triangles, so the triangle
lifting has no 2-cells. Uniform input features, random weights, no positional encoding.
"""

import numpy as np
import networkx as nx


def incidence(G):
    E = [tuple(sorted(e)) for e in G.edges()]
    B1 = np.zeros((G.number_of_nodes(), len(E)))
    for j, (u, v) in enumerate(E):
        B1[u, j], B1[v, j] = 1, 1
    return E, B1


def af3(G, E):
    return np.array([4 - G.degree(u) - G.degree(v) + 3 * len(list(nx.common_neighbors(G, u, v)))
                     for u, v in E], dtype=float)


def row_normalize(A):
    s = A.sum(1, keepdims=True)
    return A / np.where(s > 0, s, 1)


def mp(B1, kappa, aggr, gate, D=64, layers=3, seed=0):
    rng = np.random.default_rng(seed)
    W = {k: rng.normal(size=(D, D)) / np.sqrt(D) for k in ["vE", "adj", "e0", "EV", "v0"]}
    Wg = rng.normal(size=(D, D + 1)) / np.sqrt(D)
    A_ve, A_ee, A_ev = B1.T, B1.T @ B1 - 2 * np.eye(B1.shape[1]), B1
    if aggr == "mean":
        A_ve, A_ee, A_ev = row_normalize(A_ve), row_normalize(A_ee), row_normalize(A_ev)
    if gate == "standardized":
        k = (kappa - kappa.mean()) / (kappa.std() + 1e-5) if kappa.std() > 1e-6 else kappa - kappa.mean()
    elif gate == "raw":
        k = kappa
    else:
        k = np.zeros_like(kappa)
    HV, HE = np.ones((B1.shape[0], D)), np.ones((B1.shape[1], D))
    for _ in range(layers):
        Ht = np.tanh((HE + (A_ve @ HV) @ W["vE"].T + (A_ee @ HE) @ W["adj"].T) @ W["e0"].T)
        g = 1 / (1 + np.exp(-np.hstack([Ht, k[:, None]]) @ Wg.T)) if gate != "none" else 1.0
        HE = Ht * g
        HV = np.tanh((HV + (A_ev @ HE) @ W["EV"].T) @ W["v0"].T)
    readout = (lambda X: X.mean(0)) if aggr == "mean" else (lambda X: X.sum(0))
    return np.concatenate([readout(HV), readout(HE)])


if __name__ == "__main__":
    graphs = {"C8": nx.cycle_graph(8), "Q3": nx.convert_node_labels_to_integers(nx.hypercube_graph(3))}
    data = {}
    for name, G in graphs.items():
        E, B1 = incidence(G)
        data[name] = (B1, af3(G, E))
        print(f"{name}: |V|={G.number_of_nodes()} |E|={len(E)} AF3 values={sorted(set(data[name][1]))}")
    print()
    for aggr in ["sum", "mean"]:
        for gate in ["none", "standardized", "raw"]:
            h = {n: mp(B1, k, aggr, gate) for n, (B1, k) in data.items()}
            d = np.linalg.norm(h["C8"] - h["Q3"])
            print(f"aggregation/readout={aggr:4s}  gate={gate:12s}  ||h(C8)-h(Q3)||_2 = {d:.3e}")
