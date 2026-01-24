# src/clustering/behavior/predict.py

import os
import sys
import joblib
import pandas as pd
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

from src.clustering.behavior.vectorizer import BehaviorVectorizer
from src.config.settings import BEHAVIOR_CLUSTER_MODEL_PATH


class BehaviorClusterPredictor:
    """
    Predictor cho behavior-based user clustering.

    - Load sẵn model & scaler
    - Validate input
    - Handle cold-start user
    - Trả về mapping user_id → behavior_cluster
    """

    REQUIRED_FEATURES = [
        "total_products",
        "unique_products",
        "reorder_ratio"
    ]

    def __init__(self, default_cluster: int = 0):
        if not os.path.exists(BEHAVIOR_CLUSTER_MODEL_PATH):
            raise FileNotFoundError(
                "Behavior clustering model not found. Please train the model first."
            )

        self.model = joblib.load(BEHAVIOR_CLUSTER_MODEL_PATH)

        self.vectorizer = BehaviorVectorizer()
        self.vectorizer.load()

        self.default_cluster = default_cluster

    def _validate_input(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        missing = set(self.REQUIRED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required behavior features: {missing}")

    def _handle_cold_start(self, df: pd.DataFrame) -> pd.Series:
        """
        Cold-start heuristic:
        - Nếu user chưa có lịch sử mua → gán cluster mặc định
        """
        is_cold = (
            (df["total_products"] == 0) &
            (df["unique_products"] == 0)
        )
        return is_cold

    def predict(self, df_user: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_user : pd.DataFrame
            Yêu cầu có cột:
            - user_id
            - total_products
            - unique_products
            - reorder_ratio

        Returns
        -------
        pd.DataFrame
            user_id | behavior_cluster
        """

        self._validate_input(df_user)

        df = df_user.copy()

        if "user_id" not in df.columns:
            raise ValueError("Missing user_id column.")

        # Handle cold-start users
        cold_mask = self._handle_cold_start(df)

        # Predict for non-cold users
        df_predict = df.loc[~cold_mask]
        clusters = np.full(len(df), self.default_cluster)

        if not df_predict.empty:
            X = self.vectorizer.transform(df_predict)
            clusters[~cold_mask] = self.model.predict(X)

        # Build output
        result = pd.DataFrame({
            "user_id": df["user_id"].values,
            "behavior_cluster": clusters
        })

        return result
