import matplotlib.pyplot as plt
import numpy as np


def plot_mlp_classifier(results: list[dict]):
    hidden_sizes = sorted({r['hidden_size'] for r in results})
    learning_rates = sorted({r['learning_rate'] for r in results})

    means = np.zeros((len(hidden_sizes), len(learning_rates)))
    stds = np.zeros_like(means)
    for r in results:
        i = hidden_sizes.index(r['hidden_size'])
        j = learning_rates.index(r['learning_rate'])
        means[i, j] = r['mean']
        stds[i, j] = r['std']

    best = max(results, key=lambda r: r['mean'])
    best_hidden_idx = hidden_sizes.index(best['hidden_size'])
    best_lr_idx = learning_rates.index(best['learning_rate'])

    _, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(hidden_sizes))
    width = 0.25
    # Learning rates are already sorted ascending: [0.001, 0.01, 0.1]
    # Color palette: dark red → red → light red
    colors = ['#8B0000', '#DC143C', '#FF6666']

    for j, lr in enumerate(learning_rates):
        offset = (j - (len(learning_rates) - 1) / 2) * width
        bars = ax.bar(x + offset, means[:, j], width, yerr=stds[:, j],
                      label=f'lr={lr}', color=colors[j], capsize=3,
                      edgecolor='black', linewidth=0.5)
        if j == best_lr_idx:
            bars[best_hidden_idx].set_edgecolor('gold')
            bars[best_hidden_idx].set_linewidth(2.5)

    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Mean Balanced Accuracy')
    ax.set_title('MLPClassifier — 5-Fold Cross Validation (mean +/- std)')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_sizes)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Learning rate', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    best_label = (f'Best: hidden={best["hidden_size"]}, '
                  f'lr={best["learning_rate"]} → {best["mean"]:.2f}')
    ax.annotate(best_label, xy=(0.02, 0.98), xycoords='axes fraction',
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gold', alpha=0.9))

    plt.tight_layout()
    plt.show()
