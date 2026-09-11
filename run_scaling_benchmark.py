import time
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def benchmark_scaling():
    sizes = [100, 200, 400, 800, 1600]
    bounded_times = []
    scalefree_times = []
    
    print(f"{'Nodes (N)':>10} | {'Bounded Ring Complex (s)':>25} | {'Scale-Free Unconstrained (s)':>28}")
    print("-" * 70)
    
    for n in sizes:
        # 1. Bounded Molecular / Ring Complex (Planar ring lattice, k_max <= 6)
        side = int(np.sqrt(n))
        G_bounded = nx.grid_2d_graph(side, side)
        G_bounded = nx.convert_node_labels_to_integers(G_bounded)
        
        cycles_bounded = list(nx.chordless_cycles(G_bounded, length_bound=6))
        # Build boundary matrix B2
        edges = list(G_bounded.edges())
        edge_map = {tuple(sorted(e)): i for i, e in enumerate(edges)}
        
        B2_entries = []
        for face_idx, c in enumerate(cycles_bounded):
            c_edges = [(c[i], c[(i+1)%len(c)]) for i in range(len(c))]
            for ce in c_edges:
                key = tuple(sorted(ce))
                if key in edge_map:
                    B2_entries.append((edge_map[key], face_idx))
                    
        E_count = len(edges)
        F_count = len(cycles_bounded)
        if F_count > 0:
            i_idx = torch.tensor([e[0] for e in B2_entries], dtype=torch.long)
            j_idx = torch.tensor([e[1] for e in B2_entries], dtype=torch.long)
            v = torch.ones(len(B2_entries), dtype=torch.float32)
            B2_sparse = torch.sparse_coo_tensor(torch.stack([i_idx, j_idx]), v, (E_count, F_count)).to_dense()
            
            start = time.time()
            L_F = torch.matmul(B2_sparse, B2_sparse.t())
            t_bounded = time.time() - start
        else:
            t_bounded = 0.0
        bounded_times.append(t_bounded)
        
        # 2. Scale-Free Barabasi-Albert Network (m=3) with hub clustering
        G_ba = nx.barabasi_albert_graph(n, 3, seed=42)
        cycles_ba = list(nx.chordless_cycles(G_ba, length_bound=6))[:25000]
        edges_ba = list(G_ba.edges())
        edge_map_ba = {tuple(sorted(e)): i for i, e in enumerate(edges_ba)}
        
        B2_ba_entries = []
        for face_idx, c in enumerate(cycles_ba):
            c_edges = [(c[i], c[(i+1)%len(c)]) for i in range(len(c))]
            for ce in c_edges:
                key = tuple(sorted(ce))
                if key in edge_map_ba:
                    B2_ba_entries.append((edge_map_ba[key], face_idx))
                    
        E_ba = len(edges_ba)
        F_ba = len(cycles_ba)
        if F_ba > 0:
            i_idx = torch.tensor([e[0] for e in B2_ba_entries], dtype=torch.long)
            j_idx = torch.tensor([e[1] for e in B2_ba_entries], dtype=torch.long)
            v = torch.ones(len(B2_ba_entries), dtype=torch.float32)
            B2_ba_sparse = torch.sparse_coo_tensor(torch.stack([i_idx, j_idx]), v, (E_ba, F_ba)).to_dense()
            
            start = time.time()
            L_F_ba = torch.matmul(B2_ba_sparse, B2_ba_sparse.t())
            t_ba = time.time() - start
        else:
            t_ba = 0.0
        scalefree_times.append(t_ba)
        
        print(f"{n:10d} | {t_bounded:25.6f}s | {t_ba:28.6f}s")
        sys.stdout.flush()
        
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, bounded_times, 'b-o', label=r'Bounded Complex ($k_{\max} \leq 6$)', linewidth=2.2, markersize=7)
    plt.plot(sizes, scalefree_times, 'r--s', label=r'Scale-Free (Hub Proliferation)', linewidth=2.2, markersize=7)
    plt.xlabel('Number of Nodes ($|V|$)', fontsize=12)
    plt.ylabel('Face Diffusion Latency ($B_2 B_2^\\top$) [s]', fontsize=12)
    plt.title('Cellular Operator Scaling: Bounded vs. Scale-Free Topologies', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save to current dir and parent dir
    plt.tight_layout()
    plt.savefig('fig_scaling_benchmark.png', dpi=300)
    plt.savefig('../fig_scaling_benchmark.png', dpi=300)
    print("Saved updated fig_scaling_benchmark.png")

if __name__ == "__main__":
    benchmark_scaling()
