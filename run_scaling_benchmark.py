import time
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import toponetx as tnx
from train import get_incidence_matrices

def test_scaling():
    sizes = [100, 200, 400, 800, 1600]
    capped_times = []
    uncapped_times = []
    
    print(f"{'Nodes':>6} | {'Capped (k=6) MP Time':>20} | {'Uncapped MP Time':>20}")
    print("-" * 55)
    
    for n in sizes:
        G = nx.barabasi_albert_graph(n, 3, seed=42)
        
        # Capped Complex
        CC_capped = tnx.CellComplex(G)
        cycles_capped = list(nx.chordless_cycles(G, length_bound=6))
        for c in cycles_capped[:2000]:
            if len(c) > 2: CC_capped.add_cell(c, rank=2)
            
        B1_c, B2_c = get_incidence_matrices(CC_capped)
        if B2_c.is_sparse: B2_c = B2_c.to_dense()
        
        start = time.time()
        # Simulate Face message passing: H_F_new = B2^T * B2 * H_F
        if B2_c.shape[1] > 0:
            L_F = torch.matmul(B2_c, B2_c.t())
        t_capped = time.time() - start
        capped_times.append(t_capped)
        
        # Uncapped Complex (Fundamental Cycle Basis)
        if n <= 800:
            CC_uncapped = tnx.CellComplex(G)
            # Find cycle basis which generates $e - n + 1$ cycles
            cycles_uncapped = nx.cycle_basis(G)
            for c in cycles_uncapped[:2000]:
                if len(c) > 2: CC_uncapped.add_cell(c, rank=2)
                
            B1_u, B2_u = get_incidence_matrices(CC_uncapped)
            if B2_u.is_sparse: B2_u = B2_u.to_dense()
            
            start = time.time()
            if B2_u.shape[1] > 0:
                L_F = torch.matmul(B2_u, B2_u.t())
            t_uncapped = time.time() - start
        else:
            t_uncapped = float('nan')
            
        uncapped_times.append(t_uncapped)
        
        print(f"{n:6d} | {t_capped:19.4f}s | {t_uncapped:19.4f}s")
        sys.stdout.flush()
        
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, capped_times, 'b-o', label='Capped B2 (k=6)', linewidth=2)
    plt.plot(sizes[:len(uncapped_times)], uncapped_times, 'r--x', label='Uncapped B2 (Basis)', linewidth=2)
    plt.xlabel('Number of Nodes ($N$)')
    plt.ylabel('Message Passing Time ($B_2 B_2^T$)')
    plt.title('Cellular Message Passing Scaling: Capped vs Uncapped')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig_scaling_benchmark.png')
    print("Saved plot.")

if __name__ == "__main__":
    test_scaling()
