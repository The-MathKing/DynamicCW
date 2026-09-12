"""
Vector Figure Generation Script for Curvature-Gated Cellular Message Passing Paper.
Generates publication-quality figures directly from JSON logs:
- Figure 1: Discrete Forman-Ricci Curvature on Molecular Graph with Bottleneck Annotations.
- Figure 2: Empirical Ablation Benchmark (Test MAE, Mean +- 95% CI across 10 seeds).
- Figure 3: Dirichlet Energy Decay Trajectory across Cellular Layers (Normalized Hodge 0- and 1-Energies).
- Figure 4: Controlled Over-Squashing and Bottleneck Transfer Accuracy.
"""
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

# Use high-quality matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

os.makedirs('figures', exist_ok=True)
os.makedirs('diagrams', exist_ok=True)

def generate_ablation_plot():
    results_path = 'results/comprehensive_ablation_results.json'
    if not os.path.exists(results_path):
        print(f"Skipping ablation plot: {results_path} not found yet.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    models = list(results.keys())
    means = [results[m]['mean'] for m in models]
    ci95s = [results[m]['ci95'] for m in models]
    
    # Sort for visual clarity
    sorted_indices = np.argsort(means)
    models_sorted = [models[i] for i in sorted_indices]
    means_sorted = [means[i] for i in sorted_indices]
    ci95s_sorted = [ci95s[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    y_pos = np.arange(len(models_sorted))
    
    colors = []
    for m in models_sorted:
        if 'GIN Baseline' in m:
            colors.append('#2b5c8f')
        elif 'AF3 Gated' in m:
            colors.append('#d95f02')
        elif 'Degree-Only' in m:
            colors.append('#7570b3')
        elif 'Cycle-Aware' in m:
            colors.append('#e7298a')
        elif 'Shuffled' in m or 'No Gate' in m:
            colors.append('#66a61e')
        else:
            colors.append('#e6ab02')
            
    bars = ax.barh(y_pos, means_sorted, xerr=ci95s_sorted, align='center', 
                   color=colors, alpha=0.85, edgecolor='black', capsize=4, height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_sorted)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('ZINC-12k Test MAE (Lower is Better, 10 Seeds $\pm$ 95% CI)')
    ax.set_title('Ablation Analysis of Cellular Message Passing and Curvature Gating')
    
    for i, (m, v, c) in enumerate(zip(models_sorted, means_sorted, ci95s_sorted)):
        ax.text(v + c + 0.005, i, f"{v:.4f}", va='center', fontsize=9.5, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('figures/fig2_ablation_grid.png', dpi=300)
    plt.savefig('figures/fig2_ablation_grid.pdf')
    print("Saved figures/fig2_ablation_grid.png and .pdf")

def generate_dirichlet_plot():
    results_path = 'results/dirichlet_energy_results.json'
    if not os.path.exists(results_path):
        print(f"Skipping dirichlet plot: {results_path} not found yet.")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    layers = np.arange(len(list(data.values())[0]['mean_node_energy']))
    
    markers = {'DynamicCW (With Residuals & Norm)': 'o', 
               'DynamicCW (Unregularized, No Residuals)': 's',
               'DynamicCW (No Gate, With Residuals)': '^'}
    colors = {'DynamicCW (With Residuals & Norm)': '#d95f02', 
              'DynamicCW (Unregularized, No Residuals)': '#7570b3',
              'DynamicCW (No Gate, With Residuals)': '#2b5c8f'}
              
    for name, vals in data.items():
        m_e0 = vals['mean_node_energy']
        std_e0 = vals['std_node_energy']
        m_e1 = vals['mean_edge_energy']
        std_e1 = vals['std_edge_energy']
        
        ax1.plot(layers, m_e0, label=name, marker=markers.get(name, 'o'), color=colors.get(name, 'black'), lw=2)
        ax1.fill_between(layers, np.maximum(1e-4, np.array(m_e0) - np.array(std_e0)), np.array(m_e0) + np.array(std_e0), alpha=0.15, color=colors.get(name, 'black'))

        ax2.plot(layers, m_e1, label=name, marker=markers.get(name, 'o'), color=colors.get(name, 'black'), lw=2)
        ax2.fill_between(layers, np.maximum(0, np.array(m_e1) - np.array(std_e1)), np.array(m_e1) + np.array(std_e1), alpha=0.15, color=colors.get(name, 'black'))

    ax1.set_xlabel('Cellular Layer $\ell$')
    ax1.set_ylabel('Normalized 0-Dirichlet Energy $\mathcal{E}_0(H_V^{(\ell)})$')
    ax1.set_title('0-Cell (Node) Dirichlet Energy')
    ax1.set_yscale('log')
    ax1.legend(frameon=True, fontsize=9)

    ax2.set_xlabel('Cellular Layer $\ell$')
    ax2.set_ylabel('Normalized 1-Dirichlet Energy $\mathcal{E}_1(H_E^{(\ell)})$')
    ax2.set_title('1-Cell (Edge) Dirichlet Energy')
    
    plt.tight_layout()
    plt.savefig('figures/fig3_dirichlet_energy.png', dpi=300)
    plt.savefig('figures/fig3_dirichlet_energy.pdf')
    print("Saved figures/fig3_dirichlet_energy.png and .pdf")

if __name__ == '__main__':
    generate_ablation_plot()
    generate_dirichlet_plot()
