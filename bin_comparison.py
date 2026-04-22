from pathlib import Path

import pandas as pd

from models.linear_regression import LinearRegressionClassifier
from models.mlp_classifier import MLPClassifierModel
from preprocess import BIN_CONFIGS, preprocess, processed_data_path


RAW_DATA_PATH = 'data/housing_price_data.csv'


def _ensure_data_for_key(key: int):
    path = Path(processed_data_path(key))
    if path.exists():
        return
    print(f'\nPreprocessed data not found for key {key}. Generating...\n')
    df = preprocess(RAW_DATA_PATH, key=key)
    df.to_csv(path, index=False)


def run_bin_comparison() -> list[dict]:
    results = []
    for key, cfg in BIN_CONFIGS.items():
        print(f'\n {"_"*10} Evaluating bin config {key}: '
              f'{cfg["lower_name"]}M / {cfg["upper_name"]}M {"_"*10} \n')
        _ensure_data_for_key(key)
        df = pd.read_csv(processed_data_path(key))

        lr = LinearRegressionClassifier(n_splits=5, random_state=42)
        lr_result = lr.evaluate(df)

        mlp = MLPClassifierModel(
            hidden_sizes=[20, 40, 60, 80, 100],
            learning_rates=[0.001, 0.01, 0.1],
            n_splits=5,
            random_state=42,
        )
        mlp_all = mlp.evaluate(df)
        mlp_best = max(mlp_all, key=lambda r: r['mean'])

        results.append({
            'key': key,
            'label': f'{cfg["lower_name"]}M / {cfg["upper_name"]}M',
            'lr_mean': float(lr_result['mean']),
            'lr_std': float(lr_result['std']),
            'mlp_mean': float(mlp_best['mean']),
            'mlp_std': float(mlp_best['std']),
            'mlp_best_config': (mlp_best['hidden_size'], mlp_best['learning_rate']),
        })
    return results


if __name__ == '__main__':
    from plots.bin_comparison_plot import plot_bin_comparison

    data = run_bin_comparison()

    print(f'\n {"_"*10} Summary {"_"*10}')
    for r in data:
        print(f'  {r["label"]:>15}  |  LR: {r["lr_mean"]:.4f} (+/- {r["lr_std"]:.4f})  '
              f'|  MLP: {r["mlp_mean"]:.4f} (+/- {r["mlp_std"]:.4f})  '
              f'[hidden={r["mlp_best_config"][0]}, lr={r["mlp_best_config"][1]}]')

    plot_bin_comparison(data)
