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

from src.recommendation.hybrid_recommender import (
    HybridRecommender,
    RULE,
    POPULAR,
    SIMILAR_DEPT,
    INSURANCE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OfflineEvaluator:
    """
    Offline evaluation for Hybrid Recommender using prior/train split.

    Metrics:
        - Precision@K
        - Recall@K
        - HitRate@K
        - UserCoverage
        - Source Coverage Metrics (RULE / POPULAR / INSURANCE / SIMILAR_DEPT)
    """

    def __init__(
        self,
        recommender: HybridRecommender,
        max_users: Optional[int] = None
    ):
        self.recommender = recommender
        self.max_users = max_users

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

        logger.info(
            f"OfflineEvaluator initialized with {len(self.user_ground_truth)} users."
        )

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

        precisions, recalls, hit_rates = [], [], []
        user_recommendations: Dict[int, List[str]] = {}

        # -------- Source coverage counters --------
        users_with_rule = 0
        total_recommended_items = 0
        rule_item_count = 0

        slot_rule_users = defaultdict(set)   # position -> set(user_id)
        hit_source_count = defaultdict(int)

        evaluated_users = 0

        for user_id, ground_truth in self.user_ground_truth.items():
            history = self.user_history.get(user_id, [])
            if not history:
                continue

            recs, _ = self.recommender.recommend(
                user_id=user_id,
                basket=history[-5:],
                time_bucket="unknown",
                is_weekend=False,
                top_k=k,
                return_metadata=True,
            )

            if not recs:
                user_recommendations[user_id] = []
                continue

            evaluated_users += 1

            # -------- User-level rule coverage --------
            if any(str(RULE) in r["source"] for r in recs):
                users_with_rule += 1

            # -------- Item & slot coverage --------
            for idx, r in enumerate(recs):
                total_recommended_items += 1
                sources = r.get("source", [])

                if str(RULE) in sources:
                    rule_item_count += 1
                    slot_rule_users[idx + 1].add(user_id)

                # -------- Hit source (primary only) --------
                if str(r["item_id"]) in ground_truth and sources:
                    primary_source = sources[0]
                    hit_source_count[primary_source] += 1

            recommended_items = [str(r["item_id"]) for r in recs]
            user_recommendations[user_id] = recommended_items

            precisions.append(precision_at_k(recommended_items, ground_truth, k))
            recalls.append(recall_at_k(recommended_items, ground_truth, k))
            hit_rates.append(hit_rate_at_k(recommended_items, ground_truth, k))

        if evaluated_users == 0:
            logger.warning("No users evaluated.")
            return {}

        # ============================================================
        # Final metrics
        # ============================================================

        metrics = {
            f"Precision@{k}": sum(precisions) / evaluated_users,
            f"Recall@{k}": sum(recalls) / evaluated_users,
            f"HitRate@{k}": sum(hit_rates) / evaluated_users,
            "UserCoverage": user_coverage(user_recommendations),
            "RuleUserCoverage": users_with_rule / evaluated_users,
            "RuleItemShare": rule_item_count / max(1, total_recommended_items),
            "num_users_evaluated": evaluated_users,
        }

        # Slot coverage
        for pos in [1, 3, 5]:
            metrics[f"RuleSlot@{pos}"] = (
                len(slot_rule_users[pos]) / evaluated_users
            )

        # Hit source share
        total_hits = sum(hit_source_count.values())
        if total_hits > 0:
            for src, cnt in hit_source_count.items():
                metrics[f"{src}_HitShare"] = cnt / total_hits

        logger.info("Offline evaluation completed")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {save_path}")

        return metrics
