import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

def generate_ablation_plots(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    color_dyn = '#c0392b'   # Crimson for Dynamic
    color_stat = '#f39c12'  # Orange for Static
    color_gin = '#2980b9'   # Blue for GIN
    
    # ---------------------------------------------------
    # Plot 1: Test Accuracy Convergence
    # ---------------------------------------------------
    ax1.plot(df['Epoch'], df['DynamicCW_Test_Acc_Mean'], label='DynamicCW-Net (Ours)', color=color_dyn)
    ax1.plot(df['Epoch'], df['StaticCW_Test_Acc_Mean'], label='StaticCW-Net (Ablation)', color=color_stat, linestyle='-.')
    ax1.plot(df['Epoch'], df['GIN_Test_Acc_Mean'], label='GIN Baseline (1-WL)', color=color_gin, linestyle='--')
    
    ax1.set_title('Test Accuracy Ablation on NCI1 (5-Fold Mean)')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    final_epoch = df['Epoch'].iloc[-1]
    final_dyn_acc = df['DynamicCW_Test_Acc_Mean'].iloc[-1]
    final_stat_acc = df['StaticCW_Test_Acc_Mean'].iloc[-1]
    final_gin_acc = df['GIN_Test_Acc_Mean'].iloc[-1]
    
    y_min, y_max = ax1.get_ylim()
    ax1.set_ylim(y_min, max(y_max, final_dyn_acc + 4.0))

    ax1.annotate(f'+{final_dyn_acc - final_stat_acc:.2f}% (Dynamic vs Static)', 
                 xy=(final_epoch, final_dyn_acc), 
                 xytext=(final_epoch - 60, final_dyn_acc + 2.0),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=11, fontweight='bold', color=color_dyn)

    ax1.annotate(f'+{final_stat_acc - final_gin_acc:.2f}% (Static vs GIN)', 
                 xy=(final_epoch, final_stat_acc), 
                 xytext=(final_epoch - 60, final_stat_acc - 2.0),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                 fontsize=11, fontweight='bold', color=color_stat)

    # ---------------------------------------------------
    # Plot 2: Training Loss Optimization
    # ---------------------------------------------------
    ax2.plot(df['Epoch'], df['DynamicCW_Train_Loss_Mean'], label='DynamicCW-Net (Ours)', color=color_dyn)
    ax2.plot(df['Epoch'], df['StaticCW_Train_Loss_Mean'], label='StaticCW-Net (Ablation)', color=color_stat, linestyle='-.')
    ax2.plot(df['Epoch'], df['GIN_Train_Loss_Mean'], label='GIN Baseline (1-WL)', color=color_gin, linestyle='--')
    
    ax2.set_title('Training Loss Ablation Trajectory')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('NCI1_Ablation_Analysis.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('NCI1_Ablation_Analysis.png', format='png', bbox_inches='tight', dpi=300)
    print("Successfully generated NCI1_Ablation_Analysis.pdf and .png")

if __name__ == "__main__":
    generate_ablation_plots('ablation_loss_curves.csv')
