import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class MLPClassifierModel:
    def __init__(
        self,
        hidden_sizes: list[int] = [20, 40, 60, 80, 100],
        learning_rates: list[float] = [0.001, 0.01, 0.1],
        n_splits: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.learning_rates = learning_rates
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.random_state = random_state

    def _evaluate_config(self, X: np.ndarray, y: np.ndarray, hidden_size: int, lr: float) -> dict:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            # StandardScaler inside the pipeline — MLPs need scaled inputs to converge
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(hidden_size,),
                    learning_rate_init=lr,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            fold_scores.append(balanced_accuracy_score(y[test_idx], y_pred))

        fold_scores = np.array(fold_scores)
        return {
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'fold_scores': fold_scores,
            'mean': fold_scores.mean(),
            'std': fold_scores.std(),
        }

    def evaluate(self, df: pd.DataFrame, target_col: str = 'price_category') -> list[dict]:
        X = df.drop(columns=[target_col]).to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=int)

        results = []
        for hidden_size in self.hidden_sizes:
            for lr in self.learning_rates:
                res = self._evaluate_config(X, y, hidden_size, lr)
                print(f'  hidden={hidden_size:3d}, lr={lr:<6}: '
                      f'mean balanced accuracy = {res["mean"]:.4f}')
                results.append(res)

        return results
