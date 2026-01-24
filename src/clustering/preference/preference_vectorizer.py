# src/clustering/preference/vectorizer.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import normalize
from src.config.settings import PREFERENCE_SCALER_PATH


class PreferenceVectorizer:
    """
    Vector hóa preference user–category
    - Normalize theo user (L1)
    - Phù hợp cho cosine-based clustering
    """

    def __init__(self, norm: str = "l1"):
        self.norm = norm
        self.feature_cols = None

    def fit_transform(self, df: pd.DataFrame):
        self._validate_input(df)

        self.feature_cols = [c for c in df.columns if c != "user_id"]
        X = df[self.feature_cols].values

        X_norm = normalize(X, norm=self.norm)
        return X_norm

    def transform(self, df: pd.DataFrame):
        self._validate_input(df, check_cols=True)

        X = df[self.feature_cols].values
        return normalize(X, norm=self.norm)

    def save(self):
        os.makedirs(os.path.dirname(PREFERENCE_SCALER_PATH), exist_ok=True)
        joblib.dump(
            {
                "norm": self.norm,
                "feature_cols": self.feature_cols
            },
            PREFERENCE_SCALER_PATH
        )

    def load(self):
        obj = joblib.load(PREFERENCE_SCALER_PATH)
        self.norm = obj["norm"]
        self.feature_cols = obj["feature_cols"]

    def _validate_input(self, df: pd.DataFrame, check_cols: bool = False):
        if "user_id" not in df.columns:
            raise ValueError("Preference data must contain user_id")

        if check_cols and self.feature_cols:
            missing = set(self.feature_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing preference features: {missing}")
