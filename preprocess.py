import pandas as pd


# Price-bin configurations explored during preprocessing.
# lower_num / upper_num are raw dollar thresholds used by bin_price.
# lower_name / upper_name are the same values in 'M' units for plot text.
BIN_CONFIGS = {
    0: {'lower_num': 5_000_000, 'upper_num': 10_000_000, 'lower_name': 5,   'upper_name': 10},
    1: {'lower_num': 2_500_000, 'upper_num': 5_000_000,  'lower_name': 2.5, 'upper_name': 5},
    2: {'lower_num': 4_000_000, 'upper_num': 6_000_000,  'lower_name': 4,   'upper_name': 6},
    3: {'lower_num': 3_500_000, 'upper_num': 5_500_000,  'lower_name': 3.5, 'upper_name': 5.5},
    4: {'lower_num': 4_000_000, 'upper_num': 5_500_000,  'lower_name': 4,   'upper_name': 5.5},
}

# Active bin configuration - change this number to switch thresholds everywhere.
CLASS_THRESHOLD_KEY = 4


def processed_data_path(key: int) -> str:
    return f'data/{key}_housing_price_processed.csv'


def preprocess(filepath: str, key: int | None = None) -> pd.DataFrame:
    # 1. Load dataset
    df = pd.read_csv(filepath)

    # 2. Binary encoding: Yes -> 1, No -> 0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_cols] = df[binary_cols].apply(lambda col: col.map({'yes': 1, 'no': 0}))

    # 3. One-hot encoding for furnishingstatus
    df = pd.get_dummies(df, columns=['furnishingstatus'], prefix='furnishingstatus')

    # 4. Target label binning: price -> class 0, 1, or 2
    bin_key = key if key is not None else CLASS_THRESHOLD_KEY
    active = BIN_CONFIGS[bin_key]
    lower, upper = active['lower_num'], active['upper_num']

    def bin_price(price):
        if price <= lower:
            return 0
        elif price <= upper:
            return 1
        else:
            return 2

    df['price_category'] = df['price'].apply(bin_price)
    df = df.drop(columns=['price'])

    return df


if __name__ == "__main__":
    raw_path = 'data/housing_price_data.csv'
    for k in BIN_CONFIGS:
        data_frame = preprocess(raw_path, key=k)
        out_path = processed_data_path(k)
        data_frame.to_csv(out_path, index=False)
        cfg = BIN_CONFIGS[k]
        print(f'\n--- Key {k}: thresholds {cfg["lower_name"]}M / {cfg["upper_name"]}M → {out_path} ---')
        print('Shape:', data_frame.shape)
        print('Price category distribution:\n', data_frame['price_category'].value_counts().sort_index())
