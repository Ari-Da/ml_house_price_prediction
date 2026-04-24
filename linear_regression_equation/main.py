# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.linear_model import Ridge

from plots import plot_predicted_vs_actual, print_equation

# %%
DATA_PATH = Path(__file__).parent.parent / 'data' / '4_housing_price_processed.csv'
TARGET = 'price_category'

# %%
def main():
    # %%
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET]).to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    feature_names = [c for c in df.columns if c != TARGET]

    model = Ridge()
    model.fit(X, y)
    y_pred_cont = model.predict(X)

    # %%
    print_equation(model.coef_, model.intercept_, feature_names)

    # %%
    plot_predicted_vs_actual(y.astype(int), y_pred_cont)


if __name__ == "__main__":
    main()
