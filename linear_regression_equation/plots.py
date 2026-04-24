import matplotlib.pyplot as plt
import numpy as np


def print_equation(coefs, intercept, feature_names):
    print('Fitted Ridge equation (all features):')
    print(f'  intercept (b) = {intercept:+.6f}')
    for w, name in zip(coefs, feature_names):
        print(f'    {w:+.6e}  ·  {name}')
    print()


def plot_predicted_vs_actual(y_true, y_pred_cont):
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(y_true))
    x = y_true + jitter

    y_pred_class = np.where(y_pred_cont < 0.5, 0, np.where(y_pred_cont < 1.5, 1, 2))
    correct = y_pred_class == y_true
    wrong = ~correct

    n_correct = int(correct.sum())
    n_wrong = int(wrong.sum())
    total = len(correct)

    _, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[correct], y_pred_cont[correct],
               c='#3C8D40', alpha=0.6, edgecolors='black', linewidths=0.4,
               label=f'Correctly classified: {n_correct} / {total} ({n_correct / total:.1%})')
    ax.scatter(x[wrong], y_pred_cont[wrong],
               c='#B8322C', alpha=0.6, edgecolors='black', linewidths=0.4,
               label=f'Misclassified: {n_wrong} / {total} ({n_wrong / total:.1%})')

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(1.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(-0.5, 2.6)
    ax.text(2.38, 0.5, 'threshold 0.5', va='center', color='gray', fontsize=9)
    ax.text(2.38, 1.5, 'threshold 1.5', va='center', color='gray', fontsize=9)

    ax.set_xticks([0, 1, 2])
    ax.set_xlabel('True class (jittered horizontally for visibility)')
    ax.set_ylabel('Predicted continuous ŷ')
    ax.set_title('Linear Regression — Predicted Output vs True Class')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.show()
