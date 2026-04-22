import pandas as pd

def preprocess(filepath: str) -> pd.DataFrame:
    # 1. Load dataset
    df = pd.read_csv(filepath)

    # 2. Binary encoding: Yes -> 1, No -> 0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_cols] = df[binary_cols].apply(lambda col: col.map({'yes': 1, 'no': 0}))

    # 3. One-hot encoding for furnishingstatus
    df = pd.get_dummies(df, columns=['furnishingstatus'], prefix='furnishingstatus')

    # 4. Target label binning: price -> class 0, 1, or 2
    def bin_price(price):
        if price <= 5_000_000:
            return 0
        elif price <= 10_000_000:
            return 1
        else:
            return 2

    df['price_category'] = df['price'].apply(bin_price)
    df = df.drop(columns=['price'])

    return df


if __name__ == "__main__":
    data_frame = preprocess('data/housing_price_data.csv')
    data_frame.to_csv('data/housing_price_process_data.csv', index=False)
    print(data_frame.head())
    print('\nColumns:', data_frame.columns.tolist())
    print('Shape:', data_frame.shape)
    print('\nPrice category distribution:\n', data_frame['price_category'].value_counts().sort_index())
