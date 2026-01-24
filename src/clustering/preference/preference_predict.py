# src/clustering/preference/predict.py
import os
import sys
import joblib
import pandas as pd

# Ensure project root in PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_dir)

from src.clustering.preference.vectorizer import PreferenceVectorizer
from src.config.settings import (
    PREFERENCE_CLUSTER_MODEL_PATH,
    PREFERENCE_SCALER_PATH
)


class PreferenceClusterPredictor:
    """
    Predict preference cluster cho user dựa trên vector sở thích sản phẩm

    Input:
        df_user: DataFrame
            - Mỗi dòng = 1 user
            - Các cột = feature numeric giống hệt file preference_features lúc train
            - KHÔNG bao gồm user_id
    """

    def __init__(self):
        # ---- Load model ----
        if not os.path.exists(PREFERENCE_CLUSTER_MODEL_PATH):
            raise FileNotFoundError(
                f"Preference clustering model not found: {PREFERENCE_CLUSTER_MODEL_PATH}"
            )

        if not os.path.exists(PREFERENCE_SCALER_PATH):
            raise FileNotFoundError(
                f"Preference scaler not found: {PREFERENCE_SCALER_PATH}"
            )

        self.model = joblib.load(PREFERENCE_CLUSTER_MODEL_PATH)

        # ---- Load vectorizer ----
        self.vectorizer = PreferenceVectorizer()
        self.vectorizer.load()

    def predict(self, df_user: pd.DataFrame):
        """
        Predict preference cluster

        Parameters
        ----------
        df_user : pd.DataFrame
            DataFrame chứa đúng các feature đã dùng khi train

        Returns
        -------
        np.ndarray
            Mảng cluster_id
        """

        # ---- Feature validation ----
        missing_cols = set(self.vectorizer.feature_cols) - set(df_user.columns)
        if missing_cols:
            raise ValueError(
                f"Missing features for prediction: {missing_cols}"
            )

        # Ensure column order
        df_user = df_user[self.vectorizer.feature_cols]

        # ---- Transform & Predict ----
        X = self.vectorizer.transform(df_user)
        return self.model.predict(X)
