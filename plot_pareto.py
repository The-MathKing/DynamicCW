import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality formatting
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

def plot_pareto_frontier(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'GIN Baseline (1-WL)': '#2980b9', # Blue
        'DynamicCW-Net (Ours)': '#c0392b', # Crimson
        '3-WL Graph Kernel': '#7f8c8d' # Grey
    }
    
    markers = {
        'GIN Baseline (1-WL)': 's', # Square
        'DynamicCW-Net (Ours)': '*', # Star
        '3-WL Graph Kernel': '^' # Triangle
    }
    
    for i, row in df.iterrows():
        name = row['Model']
        ax.scatter(row['Latency_ms'], row['Accuracy'], 
                   color=colors[name], 
                   marker=markers[name], 
                   s=600 if name == 'DynamicCW-Net (Ours)' else 250, 
                   label=name, 
                   edgecolor='black', 
                   linewidth=1.5,
                   zorder=3)
                   
        # Annotate points
        xytext = (15, -15)
        if name == 'DynamicCW-Net (Ours)':
            xytext = (-20, 25)
        elif name == '3-WL Graph Kernel':
            xytext = (-130, -25)
            
        ax.annotate(f"{row['Accuracy']:.1f}% | {row['Latency_ms']:.1f}ms", 
                    (row['Latency_ms'], row['Accuracy']),
                    textcoords="offset points", 
                    xytext=xytext, 
                    fontsize=11,
                    fontweight='bold',
                    color=colors[name])

    # Draw Pareto Frontier curve connecting the points
    df_sorted = df.sort_values(by='Latency_ms')
    ax.plot(df_sorted['Latency_ms'], df_sorted['Accuracy'], 
            linestyle='--', color='#2c3e50', alpha=0.5, zorder=1)

    # Use Log Scale for X because latency difference is 3 orders of magnitude
    ax.set_xscale('log')
    
    ax.set_title('Efficiency vs Expressivity (Pareto Frontier)')
    ax.set_xlabel('Inference Latency (ms) [Log Scale]')
    ax.set_ylabel('NCI1 Test Accuracy (%)')
    
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('Pareto_Frontier.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Pareto_Frontier.png', format='png', dpi=300, bbox_inches='tight')
    print("Successfully generated Pareto_Frontier.pdf and Pareto_Frontier.png")

if __name__ == "__main__":
    plot_pareto_frontier('pareto_frontier_data.csv')
