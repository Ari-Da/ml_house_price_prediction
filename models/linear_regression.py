import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score


class LinearRegressionClassifier:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _get_class_label(pred: np.ndarray) -> np.ndarray:
        # For less than 0.5 output is class 0, for less than 1.5 output is class 1, otherwise class 2
        labels = np.where(pred < 0.5, 0, np.where(pred < 1.5, 1, 2))
        return labels.astype(int)

    def evaluate(self, df: pd.DataFrame, target_col: str = 'price_category') -> dict:
        X = df.drop(columns=[target_col]).to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=int)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Ridge()
            model.fit(X_train, y_train)
            y_pred_cont = model.predict(X_test)
            y_pred = self._get_class_label(y_pred_cont)

            score = balanced_accuracy_score(y_test, y_pred)
            fold_scores.append(score)
            print(f"  Fold {fold}: balanced accuracy = {score:.4f}")

        print('\n')

        fold_scores = np.array(fold_scores)
        return {
            'fold_scores': fold_scores,
            'mean': fold_scores.mean(),
            'std': fold_scores.std(),
        }
