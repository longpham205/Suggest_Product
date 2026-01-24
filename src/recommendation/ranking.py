# src/recommendation/ranking.py

import logging
from typing import Dict, List, Tuple

from src.config.settings import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class Ranker:
    """
    Final ranking stage: aggregate multiple score sources into a single ranking.

    Score sources:
        - rule_scores       (FP-Growth, context-aware)
        - behavior_scores   (short-term behavior signal)
        - preference_scores (long-term preference)
        - lifecycle_scores  (user lifecycle adjustment)

    Responsibilities:
        - Normalize each score source independently
        - Weighted aggregation
        - Return top-K ranked candidates
    """

    # ==========================================================
    # INIT
    # ==========================================================
    def __init__(
        self,
        rule_weight: float = 0.5,
        behavior_weight: float = 0.2,
        preference_weight: float = 0.2,
        lifecycle_weight: float = 0.1,
    ):
        total = (
            rule_weight
            + behavior_weight
            + preference_weight
            + lifecycle_weight
        )
        if total <= 0:
            raise ValueError("Sum of weights must be positive")

        # Normalize weights
        self.rule_weight = rule_weight / total
        self.behavior_weight = behavior_weight / total
        self.preference_weight = preference_weight / total
        self.lifecycle_weight = lifecycle_weight / total

        logger.info(
            "Ranker initialized | "
            f"weights={{rule={self.rule_weight:.2f}, "
            f"behavior={self.behavior_weight:.2f}, "
            f"preference={self.preference_weight:.2f}, "
            f"lifecycle={self.lifecycle_weight:.2f}}}"
        )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================
    @staticmethod
    def _min_max_normalize(
        scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Min-max normalize scores to [0, 1].
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_v, max_v = min(values), max(values)

        # All scores equal → assign 1.0
        if min_v == max_v:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_v) / (max_v - min_v)
            for k, v in scores.items()
        }

    # ==========================================================
    # RANK
    # ==========================================================
    def rank(
        self,
        rule_scores: Dict[int, float],
        behavior_scores: Dict[int, float],
        preference_scores: Dict[int, float],
        lifecycle_scores: Dict[int, float],
        top_k: int = DEFAULT_TOP_K,
        return_scores: bool = False,
    ) -> List[int] | List[Tuple[int, float]]:
        """
        Aggregate all score sources and return ranked candidates.

        Args:
            rule_scores: product_id -> rule-based score
            behavior_scores: product_id -> behavior score
            preference_scores: product_id -> preference score
            lifecycle_scores: product_id -> lifecycle-adjusted score
            top_k: number of top items
            return_scores: whether to return (item, final_score)

        Returns:
            List[product_id] or List[(product_id, final_score)]
        """

        # Union all candidates
        all_items = (
            set(rule_scores)
            | set(behavior_scores)
            | set(preference_scores)
            | set(lifecycle_scores)
        )

        if not all_items:
            logger.warning("Ranker received empty candidate set")
            return []

        # Normalize each score source independently
        rule_scores_n = self._min_max_normalize(rule_scores)
        behavior_scores_n = self._min_max_normalize(behavior_scores)
        preference_scores_n = self._min_max_normalize(preference_scores)
        lifecycle_scores_n = self._min_max_normalize(lifecycle_scores)

        final_scores: Dict[int, float] = {}
        for pid in all_items:
            final_scores[pid] = (
                self.rule_weight * rule_scores_n.get(pid, 0.0)
                + self.behavior_weight * behavior_scores_n.get(pid, 0.0)
                + self.preference_weight * preference_scores_n.get(pid, 0.0)
                + self.lifecycle_weight * lifecycle_scores_n.get(pid, 0.0)
            )

        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_ranked = ranked[:top_k]

        logger.info(
            "Top-%d ranked items: %s",
            top_k,
            [(pid, round(score, 4)) for pid, score in top_ranked],
        )

        return top_ranked if return_scores else [pid for pid, _ in top_ranked]
