# src/recommendation/behavior_adjuster.py
"""
Behavior-based score adjuster (Optimized, Production-ready)

- Uses pre-trained scaler + clustering model (.pkl)
- Assigns behavior cluster in realtime
- O(1) lookup for product behavior scores
"""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.config.settings import (
    BEHAVIOR_CLUSTER_MODEL_PATH,
    BEHAVIOR_SCALER_PATH,
    BEHAVIOR_CLUSTER_SCORE_PATH,   # <-- cluster → product → score
)

logger = logging.getLogger(__name__)


class BehaviorAdjuster:
    """
    Behavior-based score adjuster

    Pipeline:
        user features
          → scaler.transform
          → clustering.predict
          → lookup cluster-product score
    """

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(self):
        # Load once
        self.scaler = joblib.load(BEHAVIOR_SCALER_PATH)
        self.cluster_model = joblib.load(BEHAVIOR_CLUSTER_MODEL_PATH)
        self.cluster_product_score: Dict[int, Dict[int, float]] = joblib.load(
            BEHAVIOR_CLUSTER_SCORE_PATH
        )

        logger.info(
            "BehaviorAdjuster loaded | "
            f"clusters={len(self.cluster_product_score)}"
        )

    # ==========================================================
    # CLUSTER ASSIGNMENT
    # ==========================================================
    def assign_cluster(self, feature_vector: List[float]) -> int:
        """
        Assign behavior cluster from raw feature vector
        """
        x = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return int(self.cluster_model.predict(x_scaled)[0])

    # ==========================================================
    # APPLY BEHAVIOR SCORE
    # ==========================================================
    def apply(
        self,
        scores: Dict[int, float],
        behavior_cluster: int,
    ) -> Dict[int, float]:
        """
        Adjust rule scores based on behavior cluster

        Args:
            scores: Dict[product_id, base_score]
            behavior_cluster: assigned cluster id

        Returns:
            Dict[product_id, adjusted_score]
        """
        if not scores:
            return {}

        product_weights = self.cluster_product_score.get(behavior_cluster)

        # Không có dữ liệu cho cluster → giữ nguyên
        if not product_weights:
            return scores

        adjusted = {}
        for pid, base_score in scores.items():
            weight = product_weights.get(pid, 1.0)
            adjusted[pid] = base_score * weight

        logger.debug(
            f"Behavior adjust | cluster={behavior_cluster} | "
            f"affected={sum(pid in product_weights for pid in scores)}"
        )

        return adjusted
