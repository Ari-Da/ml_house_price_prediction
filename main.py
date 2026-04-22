# %%
from pathlib import Path

import pandas as pd
from models.linear_regression import LinearRegressionClassifier
from models.mlp_classifier import MLPClassifierModel
from plots.data_plot import plot_class_distribution
from plots.linear_regression_plot import plot_linear_regression
from plots.mlp_classifier_plot import plot_mlp_classifier
from preprocess import preprocess
from rich.console import Console
from rich.rule import Rule
from rich.table import Table


RAW_DATA_PATH = 'data/housing_price_data.csv'
DATA_PATH = 'data/housing_price_process_data.csv'


def _ensure_preprocessed_data():
    if Path(DATA_PATH).exists():
        return
    print(f'Preprocessed data not found at {DATA_PATH}. Running preprocessing...')
    df = preprocess(RAW_DATA_PATH)
    df.to_csv(DATA_PATH, index=False)
    print(f'Preprocessed data written to {DATA_PATH}')

# %%
def main():
    # %%
    # Ensure preprocessed data exists before reading it
    _ensure_preprocessed_data()
    house_price_df = pd.read_csv(DATA_PATH)

    # %%
    # Plot the preprocessed data: price category class distribution
    plot_class_distribution(house_price_df)

    # %% [markdown]
    # # Linear Regression Training

    Console().print(Rule('Linear Regression Training', style='bold cyan'))
    lin_regression = LinearRegressionClassifier(n_splits=5, random_state=42)
    lin_regression_results = lin_regression.evaluate(house_price_df)

    # %%
    # Table view print of the fold results, best fold highlighted in green
    lin_fold_scores = lin_regression_results['fold_scores']
    best_fold_idx = int(lin_fold_scores.argmax())

    table = Table(title='Linear Regression — 5-Fold Cross Vlidation')
    table.add_column('Fold', justify='right')
    table.add_column('Balanced Accuracy', justify='right')
    for i, s in enumerate(lin_fold_scores, start=1):
        style = 'bold green' if (i - 1) == best_fold_idx else ''
        table.add_row(str(i), f'{s:.4f}', style=style)
    table.add_section()
    table.add_row('Mean', f'{lin_regression_results["mean"]:.4f}')
    table.add_row('Std',  f'{lin_regression_results["std"]:.4f}')

    Console().print(table)

    # %%
    print(f'\nMean balanced accuracy: {lin_regression_results["mean"]:.4f} '
          f'(+/- {lin_regression_results["std"]:.4f})')

    # %%
    # Plot the fold accuracies with the mean line overlaid
    plot_linear_regression(lin_regression_results)


    # %% [markdown]
    # # MLPClassifier Training

    Console().print(Rule('MLPClassifier Training', style='bold magenta'))
    mlp = MLPClassifierModel(
        hidden_sizes=[20, 40, 60, 80, 100],
        learning_rates=[0.001, 0.01, 0.1],
        n_splits=5,
        random_state=42,
    )
    mlp_results = mlp.evaluate(house_price_df)

    # %%
    # Table view of MLP grid results, grouped by hidden size then learning rate.
    # Best config (by mean accuracy) is highlighted in green.
    mlp_grouped = sorted(mlp_results, key=lambda r: (r['hidden_size'], r['learning_rate']))
    best = max(mlp_results, key=lambda r: r['mean'])

    mlp_table = Table(title='MLPClassifier — 5-Fold Cross Validation')
    mlp_table.add_column('Hidden Size', justify='right')
    mlp_table.add_column('Learning Rate', justify='right')
    mlp_table.add_column('Mean Balanced Accuracy', justify='right')
    mlp_table.add_column('Std', justify='right')
    for r in mlp_grouped:
        is_best = r is best
        style = 'bold green' if is_best else ''
        mlp_table.add_row(
            str(r['hidden_size']),
            f'{r["learning_rate"]}',
            f'{r["mean"]:.4f}',
            f'{r["std"]:.4f}',
            style=style,
        )

    Console().print(mlp_table)
    print(f'\nBest config: hidden_size={best["hidden_size"]}, '
          f'learning_rate={best["learning_rate"]}, '
          f'mean balanced accuracy = {best["mean"]:.4f} (+/- {best["std"]:.4f})')

    # %%
    # Plot all MLP configs and their accuracies
    plot_mlp_classifier(mlp_results)

# %%
if __name__ == "__main__":
    main()

# %%
