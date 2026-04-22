import matplotlib.pyplot as plt
import pandas as pd


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

    class_labels = {
        0: 'Class 0\n(≤ 5M)',
        1: 'Class 1\n(5M – 10M)',
        2: 'Class 2\n(> 10M)',
    }
    labels = [class_labels.get(c, str(c)) for c in counts.index]
    colors = ['#4B0082', '#8951A5', '#C8A2C8']

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
