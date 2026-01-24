# src/clustering/behavior/vectorizer.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config.settings import BEHAVIOR_SCALER_PATH


class BehaviorVectorizer:
    """
    Vectorizer cho behavior-based user clustering.

    Sử dụng các đặc trưng hành vi:
    - total_orders
    - avg_days_between_orders
    - total_products
    - reorder_ratio
    """

    FEATURE_COLS = [
        "total_orders",
        "avg_days_between_orders",
        "total_products",
        "reorder_ratio"
    ]

    LOG_FEATURES = [
        "total_orders",
        "total_products"
    ]

    def __init__(self):
        self.scaler = StandardScaler()

    def _validate_input(self, df: pd.DataFrame):
        missing = set(self.FEATURE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing behavior features: {missing}")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Clip outliers
        - Log transform skewed features
        """
        df = df.copy()

        for col in self.FEATURE_COLS:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

        for col in self.LOG_FEATURES:
            df[col] = np.log1p(df[col])

        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self._validate_input(df)
        df_proc = self._preprocess(df)
        X = df_proc[self.FEATURE_COLS].values
        return self.scaler.fit_transform(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._validate_input(df)
        df_proc = self._preprocess(df)
        X = df_proc[self.FEATURE_COLS].values
        return self.scaler.transform(X)

    def save(self):
        os.makedirs(os.path.dirname(BEHAVIOR_SCALER_PATH), exist_ok=True)
        joblib.dump(self.scaler, BEHAVIOR_SCALER_PATH)

    def load(self):
        self.scaler = joblib.load(BEHAVIOR_SCALER_PATH)
