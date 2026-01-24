# src/evaluation/offline_eval.py
import logging
import json
from collections import defaultdict
from typing import Dict, List, Set, Optional

import pandas as pd

from src.config.settings import (
    ORDER_PRIOR_PATH,
    ORDER_TRAIN_PATH,
    ORDERS_PATH,
    DEFAULT_TOP_K
)

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    user_coverage,
)

from src.recommendation.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OfflineEvaluator:
    """
    Offline evaluation for Hybrid Recommender using prior/train split.

    Metrics:
        - Precision@K
        - Recall@K
        - F1@K
        - HitRate@K
        - User Coverage
    """

    def __init__(
        self,
        recommender: HybridRecommender,
        max_users: Optional[int] = None
    ):
        self.recommender = recommender
        self.max_users = max_users

        # -----------------------
        # Load orders
        # -----------------------
        orders_df = pd.read_csv(ORDERS_PATH)
        orders_df.columns = orders_df.columns.str.strip()

        prior_df = pd.read_csv(ORDER_PRIOR_PATH)
        prior_df.columns = prior_df.columns.str.strip()

        train_df = pd.read_csv(ORDER_TRAIN_PATH)
        train_df.columns = train_df.columns.str.strip()

        self.prior_df = prior_df.merge(
            orders_df[["order_id", "user_id"]],
            on="order_id",
            how="left"
        )

        self.train_df = train_df.merge(
            orders_df[["order_id", "user_id"]],
            on="order_id",
            how="left"
        )

        self.user_history = self._build_user_history(self.prior_df)
        self.user_ground_truth = self._build_user_ground_truth(self.train_df)

        if self.max_users is not None:
            users = list(self.user_ground_truth.keys())[:self.max_users]
            self.user_ground_truth = {u: self.user_ground_truth[u] for u in users}
            self.user_history = {u: self.user_history.get(u, []) for u in users}

        logger.info(f"OfflineEvaluator initialized with {len(self.user_ground_truth)} users.")

    # ============================================================
    # Builders
    # ============================================================

    @staticmethod
    def _build_user_history(prior_df: pd.DataFrame) -> Dict[int, List[str]]:
        history = defaultdict(list)
        for _, row in prior_df.iterrows():
            history[row["user_id"]].append(str(row["product_id"]))
        return dict(history)

    @staticmethod
    def _build_user_ground_truth(train_df: pd.DataFrame) -> Dict[int, Set[str]]:
        truth = defaultdict(set)
        for _, row in train_df.iterrows():
            truth[row["user_id"]].add(str(row["product_id"]))
        return dict(truth)

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate(
        self,
        k: int = DEFAULT_TOP_K,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        hit_rates: List[float] = []

        user_recommendations: Dict[int, List[str]] = {}

        for user_id, ground_truth in self.user_ground_truth.items():
            history = self.user_history.get(user_id, [])
            if not history:
                continue

            recs = self.recommender.recommend(
                user_id=user_id,
                basket=history[-5:],     # last-N basket
                time_bucket="unknown",
                is_weekend=False,
                top_k=k,
            )

            if not recs:
                user_recommendations[user_id] = []
                continue

            recommended_items = [str(pid) for pid in recs]
            user_recommendations[user_id] = recommended_items

            p = precision_at_k(recommended_items, ground_truth, k)
            r = recall_at_k(recommended_items, ground_truth, k)

            precisions.append(p)
            recalls.append(r)
            hit_rates.append(hit_rate_at_k(recommended_items, ground_truth, k))

            # -------- F1@K --------
            if p + r > 0:
                f1s.append(2 * p * r / (p + r))
            else:
                f1s.append(0.0)

        n_users = len(precisions)
        if n_users == 0:
            logger.warning("No users evaluated.")
            return {}

        metrics = {
            f"Precision@{k}": sum(precisions) / n_users,
            f"Recall@{k}": sum(recalls) / n_users,
            f"F1@{k}": sum(f1s) / n_users,
            f"HitRate@{k}": sum(hit_rates) / n_users,
            "UserCoverage": user_coverage(user_recommendations),
            "num_users_evaluated": n_users,
        }

        logger.info("Offline evaluation completed")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {save_path}")

        return metrics
