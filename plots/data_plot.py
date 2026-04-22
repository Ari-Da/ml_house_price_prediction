import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import BIN_CONFIGS, CLASS_THRESHOLD_KEY


def plot_correlation_heatmap(df: pd.DataFrame):
    # Coerce booleans (one-hot columns) to int so df.corr() treats them numerically
    numeric_df = df.astype(float)
    corr = numeric_df.corr()

    _, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def plot_class_distribution(df: pd.DataFrame, target_col: str = 'price_category'):
    counts = df[target_col].value_counts().sort_index()

    active = BIN_CONFIGS[CLASS_THRESHOLD_KEY]
    class_labels = {
        0: f'Class 0\n(≤ {active["lower_name"]}M)',
        1: f'Class 1\n({active["lower_name"]}M – {active["upper_name"]}M)',
        2: f'Class 2\n(> {active["upper_name"]}M)',
    }
    labels = [class_labels.get(c, str(c)) for c in counts.index]
    colors = ['#B85C1F', '#E68A3C', '#F4C28A']

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', alpha=0.9)

    total = int(counts.sum())
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts.values) * 0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Price Category')
    ax.set_ylabel('Number of Records')
    ax.set_title(f'Price Category Distribution (n = {total})')
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_bin_distribution_comparison():
    # Hardcoded counts from prior exploration runs across different bin threshold pairs.
    # See memory: project_bin_exploration.md
    configs = [
        {'label': '5M / 10M', 'counts': [350, 187, 8]},
        {'label': '2.5M / 5M', 'counts': [32, 318, 195]},
        {'label': '4M / 6M', 'counts': [219, 209, 117]},
        {'label': '3.5M / 5.5M', 'counts': [158, 230, 157]},
        {'label': '4M / 5.5M', 'counts': [219, 169, 157]},
    ]

    x = np.arange(len(configs))
    width = 0.25
    colors = ['#B85C1F', '#E68A3C', '#F4C28A']
    class_names = ['Class 0', 'Class 1', 'Class 2']

    _, ax = plt.subplots(figsize=(12, 6))
    for j in range(3):
        offset = (j - 1) * width
        values = [c['counts'][j] for c in configs]
        bars = ax.bar(x + offset, values, width, label=class_names[j],
                      color=colors[j], edgecolor='black')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v),
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Bin thresholds (low / high)')
    ax.set_ylabel('Number of Records')
    ax.set_title('Price Bin Thresholds — Class Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([c['label'] for c in configs])
    ax.legend(title='Class')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
