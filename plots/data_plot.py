import matplotlib.pyplot as plt
import pandas as pd


def plot_class_distribution(df: pd.DataFrame, target_col: str = 'price_category'):
    counts = df[target_col].value_counts().sort_index()

    class_labels = {
        0: 'Class 0\n(≤ 5M)',
        1: 'Class 1\n(5M – 10M)',
        2: 'Class 2\n(> 10M)',
    }
    labels = [class_labels.get(c, str(c)) for c in counts.index]
    colors = ['#A8DADC', '#457B9D', '#1D3557']

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', alpha=0.9)

    total = int(counts.sum())
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts.values) * 0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Price Category')
    ax.set_ylabel('Number of Houses')
    ax.set_title(f'Price Category Distribution (n = {total})')
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
