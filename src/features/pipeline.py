import joblib
import numpy as np
import pandas as pd

from src.features.constants import (
    COLS_TO_DROP, V_FEATURES, TARGET_ENCODE_COLS,
    MEDIAN_IMPUTE_COLS, M_COLS,
)
from src.features.encoders import TargetEncoder


class FraudFeaturePipeline:
    """
    Fit on training data, transform train/test/live transactions.

    Usage:
        pipeline = FraudFeaturePipeline()
        X_train = pipeline.fit_transform(train_df, y_train)
        X_test  = pipeline.transform(test_df)
        pipeline.save('models/feature_pipeline.joblib')
    """

    def __init__(self):
        self.medians_: dict = {}
        self.card_amount_stats_: pd.DataFrame | None = None
        self.target_encoder_ = TargetEncoder(cols=TARGET_ENCODE_COLS)
        self.feature_names_: list[str] = []
        self._fitted = False

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FraudFeaturePipeline":
        self._fit_medians(df)
        self._fit_card_amount_stats(df)
        X = self._base_transform(df, is_train=True)
        self.target_encoder_.fit(X, y)
        # Record final column order after full transform
        self.feature_names_ = self.fit_transform(df, y).columns.tolist()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._base_transform(df, is_train=False)
        X = self.target_encoder_.transform(X)
        X = self._finalize(X)
        # Align to training columns — fill any missing with 0
        return X.reindex(columns=self.feature_names_, fill_value=0)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self._fit_medians(df)
        self._fit_card_amount_stats(df)
        X = self._base_transform(df, is_train=True)
        X = self.target_encoder_.fit_transform(X, y)
        X = self._finalize(X)
        self.feature_names_ = X.columns.tolist()
        self._fitted = True
        return X

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FraudFeaturePipeline":
        return joblib.load(path)

    # ──────────────────────────────────────────
    # Internal steps
    # ──────────────────────────────────────────

    def _fit_medians(self, df: pd.DataFrame) -> None:
        for col in MEDIAN_IMPUTE_COLS:
            if col in df.columns:
                self.medians_[col] = df[col].median()

    def _fit_card_amount_stats(self, df: pd.DataFrame) -> None:
        # Per-card mean and std of transaction amount — used for z-score feature
        stats = df.groupby('card1')['TransactionAmt'].agg(['mean', 'std']).rename(
            columns={'mean': 'card1_amt_mean', 'std': 'card1_amt_std'}
        )
        stats['card1_amt_std'] = stats['card1_amt_std'].fillna(1.0).clip(lower=1e-6)
        self.card_amount_stats_ = stats

    def _base_transform(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        X = df.copy()

        # 1. Drop columns
        X = X.drop(columns=[c for c in COLS_TO_DROP if c in X.columns])

        # 2. Time features
        X['hour_of_day'] = (X['TransactionDT'] // 3600) % 24
        X['day_of_week'] = (X['TransactionDT'] // 86400) % 7
        X = X.drop(columns=['TransactionDT'])

        # 3. Amount features
        X['amt_log'] = np.log1p(X['TransactionAmt'])
        X = X.merge(self.card_amount_stats_, on='card1', how='left')
        global_mean = self.card_amount_stats_['card1_amt_mean'].mean()
        global_std = self.card_amount_stats_['card1_amt_std'].mean()
        X['card1_amt_mean'] = X['card1_amt_mean'].fillna(global_mean)
        X['card1_amt_std'] = X['card1_amt_std'].fillna(global_std)
        X['amt_zscore'] = (X['TransactionAmt'] - X['card1_amt_mean']) / X['card1_amt_std']
        X = X.drop(columns=['card1_amt_mean', 'card1_amt_std', 'TransactionAmt'])

        # 4. Identity flag
        X['has_identity'] = X['id_01'].notna().astype(np.int8)

        # 5. Median imputation
        for col, median in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median)

        # 6. M-flags: T→1, F→0, NaN→-1
        for col in M_COLS:
            if col in X.columns:
                X[col] = X[col].map({'T': 1, 'F': 0}).fillna(-1).astype(np.int8)

        # 7. DeviceType: mobile→1, desktop→0, NaN→-1
        if 'DeviceType' in X.columns:
            X['device_mobile'] = X['DeviceType'].map({'mobile': 1, 'desktop': 0}).fillna(-1).astype(np.int8)
            X = X.drop(columns=['DeviceType'])

        # 8. Keep only desired V-features
        all_v = [c for c in X.columns if c.startswith('V')]
        keep_v = [c for c in V_FEATURES if c in X.columns]
        drop_v = [c for c in all_v if c not in keep_v]
        X = X.drop(columns=drop_v)

        # 9. Drop remaining high-cardinality string columns not target-encoded
        remaining_str = X.select_dtypes(include='object').columns.tolist()
        non_encoded_str = [c for c in remaining_str if c not in TARGET_ENCODE_COLS]
        X = X.drop(columns=non_encoded_str)

        return X

    def _finalize(self, X: pd.DataFrame) -> pd.DataFrame:
        # Fill any remaining NaN with 0 (V-features, id_ features)
        X = X.fillna(0)
        # Cast to float32 to halve memory
        X = X.astype(np.float32)
        return X
