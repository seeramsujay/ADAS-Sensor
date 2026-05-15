import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def set_sota_style():
    """Sets a professional, publication-ready plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

def plot_training_results(train_losses, snr_improvements, results_dir):
    """
    Generates SOTA-quality plots for training metrics.
    """
    set_sota_style()
    os.makedirs(results_dir, exist_ok=True)
    
    epochs = np.arange(len(train_losses))
    
    # 1. Training Loss Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=train_losses, label='Training Loss', color='#2E86AB', marker='o', markersize=8)
    
    # Add a subtle area under the curve
    plt.fill_between(epochs, train_losses, color='#2E86AB', alpha=0.1)
    
    plt.title('Early Fusion Model: Convergence Analysis', pad=20)
    plt.xlabel('Training Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_loss.png'), bbox_inches='tight')
    plt.close()

    # 2. SNR Enhancement Plot
    plt.figure(figsize=(10, 6))
    
    # Use a more vibrant color for enhancement
    sns.lineplot(x=epochs, y=snr_improvements, label='SNR Gain (dB)', color='#A23B72', marker='s', markersize=8)
    
    # Add a trend line or shaded region to emphasize improvement
    plt.fill_between(epochs, snr_improvements, color='#A23B72', alpha=0.1)
    
    plt.title('Radar-Gated SNR Optimization', pad=20)
    plt.xlabel('Training Epoch')
    plt.ylabel('SNR Enhancement [dB]')
    
    # Annotate the final gain
    final_gain = snr_improvements[-1]
    plt.annotate(f'Final Gain: {final_gain:.2f} dB', 
                 xy=(epochs[-1], final_gain), 
                 xytext=(epochs[-1]-2, final_gain-1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'snr_enhancement.png'), bbox_inches='tight')
    plt.close()

    print(f"SOTA plots successfully generated in {results_dir}")
