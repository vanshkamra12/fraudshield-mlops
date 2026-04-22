import numpy as np
import pandas as pd


class TargetEncoder:
    """
    Fit on train labels, replace category with its mean fraud rate.
    Unseen categories and NaN get the global mean (smoothed).
    Smoothing prevents high-variance estimates from rare categories.
    """

    def __init__(self, cols: list[str], smoothing: float = 10.0):
        self.cols = cols
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.mapping_: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        self.global_mean_ = y.mean()
        for col in self.cols:
            if col not in X.columns:
                continue
            stats = y.groupby(X[col]).agg(['sum', 'count'])
            # Smoothed estimate: blend category rate with global mean
            n = stats['count']
            smoothed = (stats['sum'] + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            if col not in X.columns:
                continue
            X[col] = X[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
