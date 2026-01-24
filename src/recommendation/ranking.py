# src/recommendation/ranking.py

import logging
from typing import Dict, List, Tuple, Union

from src.config.settings import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class Ranker:
    """
    Final ranking stage with rule-aware aggregation.

    Design principles:
    - Rule is a strong prior signal
    - Item appearing in any source must receive credit
    - Guarantee minimum rule exposure in top-K
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
        min_rule_slots: int = 1,
    ):
        total = (
            rule_weight
            + behavior_weight
            + preference_weight
            + lifecycle_weight
        )
        if total <= 0:
            raise ValueError("Sum of weights must be positive")

        self.rule_weight = rule_weight / total
        self.behavior_weight = behavior_weight / total
        self.preference_weight = preference_weight / total
        self.lifecycle_weight = lifecycle_weight / total
        self.min_rule_slots = max(0, min_rule_slots)

        logger.info(
            "Ranker initialized | "
            f"weights={{rule={self.rule_weight:.2f}, "
            f"behavior={self.behavior_weight:.2f}, "
            f"preference={self.preference_weight:.2f}, "
            f"lifecycle={self.lifecycle_weight:.2f}}}, "
            f"min_rule_slots={self.min_rule_slots}"
        )

    # ==========================================================
    # NORMALIZATION
    # ==========================================================
    @staticmethod
    def _min_max_normalize(
        scores: Dict[int, float]
    ) -> Dict[int, float]:
        if not scores:
            return {}

        values = list(scores.values())
        min_v, max_v = min(values), max(values)

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
    ) -> Union[List[int], List[Tuple[int, float]]]:

        # ------------------------------------------------------
        # Candidate union
        # ------------------------------------------------------
        all_items = (
            set(rule_scores)
            | set(behavior_scores)
            | set(preference_scores)
            | set(lifecycle_scores)
        )

        if not all_items:
            logger.warning("Ranker received empty candidate set")
            return []

        # ------------------------------------------------------
        # Normalize per source
        # ------------------------------------------------------
        rule_n = self._min_max_normalize(rule_scores)
        behavior_n = self._min_max_normalize(behavior_scores)
        preference_n = self._min_max_normalize(preference_scores)
        lifecycle_n = self._min_max_normalize(lifecycle_scores)

        # ------------------------------------------------------
        # Aggregate scores (NO phantom credit)
        # ------------------------------------------------------
        final_scores: Dict[int, float] = {}

        for pid in all_items:
            score = 0.0

            if pid in rule_n:
                score += self.rule_weight * rule_n[pid]

            if pid in behavior_n:
                score += self.behavior_weight * behavior_n[pid]

            if pid in preference_n:
                score += self.preference_weight * preference_n[pid]

            if pid in lifecycle_n:
                score += self.lifecycle_weight * lifecycle_n[pid]

            final_scores[pid] = score

        # ------------------------------------------------------
        # Initial ranking
        # ------------------------------------------------------
        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # ------------------------------------------------------
        # RULE-AWARE RE-RANKING
        # ------------------------------------------------------
        rule_ranked = [
            (pid, score)
            for pid, score in ranked
            if pid in rule_scores
        ]

        min_rule_slots = min(
            self.min_rule_slots,
            len(rule_ranked),
            top_k,
        )

        final_ranked: List[Tuple[int, float]] = []
        used = set()

        # 1️⃣ Guarantee rule exposure
        for pid, score in rule_ranked[:min_rule_slots]:
            final_ranked.append((pid, score))
            used.add(pid)

        # 2️⃣ Fill remaining slots by global ranking
        for pid, score in ranked:
            if len(final_ranked) >= top_k:
                break
            if pid in used:
                continue
            final_ranked.append((pid, score))
            used.add(pid)

        # ------------------------------------------------------
        # Safety + logging
        # ------------------------------------------------------
        if len(final_ranked) < top_k:
            logger.warning(
                "Ranker returned %d/%d items (candidates=%d)",
                len(final_ranked),
                top_k,
                len(all_items),
            )

        logger.info(
            "Top-%d ranked items: %s",
            top_k,
            [(pid, round(score, 4)) for pid, score in final_ranked],
        )

        return (
            final_ranked
            if return_scores
            else [pid for pid, _ in final_ranked]
        )
