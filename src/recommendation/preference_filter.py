# src/recommendation/preference_filter.py
"""
Preference-based score adjuster (Optimized, Production-ready)

- Uses pre-trained scaler + clustering model (.pkl)
- Assigns soft preference scores based on department affinity
- Does NOT generate or remove candidates
"""

import logging
from typing import Dict, List

import joblib
import numpy as np

from src.config.settings import (
    PREFERENCE_CLUSTER_MODEL_PATH,
    PREFERENCE_SCALER_PATH,
    PREFERENCE_CLUSTER_SCORE_PATH,   # cluster -> department -> score
)

logger = logging.getLogger(__name__)


class PreferenceFilter:
    """
    Preference-based score adjuster

    Pipeline:
        user preference features
          → scaler.transform
          → clustering.predict
          → lookup cluster-department score
    """

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(self):
        self.scaler = joblib.load(PREFERENCE_SCALER_PATH)
        self.cluster_model = joblib.load(PREFERENCE_CLUSTER_MODEL_PATH)
        self.cluster_department_score: Dict[str, Dict[str, float]] = joblib.load(
            PREFERENCE_CLUSTER_SCORE_PATH
        )

        logger.info(
            "PreferenceFilter loaded | "
            f"clusters={len(self.cluster_department_score)}"
        )

    # ==========================================================
    # CLUSTER ASSIGNMENT
    # ==========================================================
    def assign_cluster(self, feature_vector: List[float]) -> int:
        """
        Assign preference cluster from raw feature vector
        """
        x = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return int(self.cluster_model.predict(x_scaled)[0])

    # ==========================================================
    # APPLY PREFERENCE SCORE
    # ==========================================================
    def apply(
        self,
        candidates: List[int],
        preference_cluster: int,
        product_department_map: Dict[int, str],
    ) -> Dict[int, float]:
        """
        Assign preference scores to candidate items.

        Args:
            candidates: list of product_id
            preference_cluster: pre-assigned user preference cluster
            product_department_map: product_id -> department

        Returns:
            Dict[product_id, preference_score]
        """

        if not candidates:
            return {}

        dept_scores = self.cluster_department_score.get(preference_cluster)

        # Cluster not found → neutral scores
        if not dept_scores:
            logger.debug(
                f"PreferenceFilter | missing cluster={preference_cluster}"
            )
            return {pid: 0.0 for pid in candidates}

        scores: Dict[int, float] = {}
        missing_dept = 0

        for pid in candidates:
            dept = product_department_map.get(pid)

            if dept is None:
                scores[pid] = 0.0
                missing_dept += 1
            else:
                scores[pid] = dept_scores.get(dept, 0.0)

        if missing_dept > 0:
            logger.debug(
                f"PreferenceFilter | "
                f"cluster={preference_cluster} | "
                f"missing_department={missing_dept}"
            )

        return scores