import matplotlib.pyplot as plt
import numpy as np


def plot_linear_regression(results: dict):
    fold_scores = np.asarray(results['fold_scores'])
    mean = float(results['mean'])
    std = float(results['std'])
    n_folds = len(fold_scores)
    best_idx = int(fold_scores.argmax())

    _, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, n_folds + 1)

    colors = ['tab:green' if i == best_idx else '#003300' for i in range(n_folds)]
    bars = ax.bar(x, fold_scores, color=colors, alpha=0.85, edgecolor='black')

    ax.axhline(y=mean, color='tab:red', linestyle='--', linewidth=2,
               label=f'Mean = {mean:.2f} (+/- {std:.2f})')

    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Linear Regression — 5-Fold Cross Validation')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
