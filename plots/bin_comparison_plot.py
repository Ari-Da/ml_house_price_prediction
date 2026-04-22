import matplotlib.pyplot as plt
import numpy as np


def plot_bin_comparison(data: list[dict]):
    labels = [r['label'] for r in data]
    lr_means = np.array([r['lr_mean'] for r in data])
    lr_stds = np.array([r['lr_std'] for r in data])
    mlp_means = np.array([r['mlp_mean'] for r in data])
    mlp_stds = np.array([r['mlp_std'] for r in data])

    x = np.arange(len(data))
    width = 0.35

    _, ax = plt.subplots(figsize=(11, 6))
    lr_bars = ax.bar(x - width / 2, lr_means, width, yerr=lr_stds,
                     label='Linear Regression', color='#8B0000',
                     edgecolor='black', capsize=4)
    mlp_bars = ax.bar(x + width / 2, mlp_means, width, yerr=mlp_stds,
                      label='MLPClassifier (best config)', color='#DC143C',
                      edgecolor='black', capsize=4)

    for bar, v, s in zip(lr_bars, lr_means, lr_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.015,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, v, s in zip(mlp_bars, mlp_means, mlp_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.015,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Bin thresholds (low / high)')
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_title('Model Performance Across Price Bin Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
