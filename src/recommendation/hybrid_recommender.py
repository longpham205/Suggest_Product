# src/recommendation/hybrid_recommender.py

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.config.settings import DEFAULT_TOP_K
from src.recommendation.candidate_generator import CandidateGenerator
from src.recommendation.behavior_adjuster import BehaviorAdjuster
from src.recommendation.preference_filter import PreferenceFilter
from src.recommendation.lifecycle_adjuster import LifecycleAdjuster
from src.recommendation.ranking import Ranker
from src.recommendation.user_context_loader import UserContextLoader

logger = logging.getLogger(__name__)

# ==========================================================
# Source constants (IMPORTANT for coverage metrics)
# ==========================================================
RULE = "RULE"
POPULAR = "POPULAR"
SIMILAR_DEPT = "SIMILAR_DEPT"
INSURANCE = "INSURANCE"


class HybridRecommender:
    """
    Hybrid Recommendation Pipeline (SOURCE-SAFE)

    Recall order:
    1. Rule-based
    2. Popular (lifecycle / behavior / global)
    3. Basket similarity (department)
    4. Insurance recall (global popular)
    """

    def __init__(
        self,
        product_department_map: Dict[int, str],
        user_context_loader: UserContextLoader,
        popular_items_global: List[int] = None,
        popular_items_by_lifecycle: Dict[str, List[int]] = None,
        popular_items_by_behavior: Dict[int, List[int]] = None,
        popular_items_by_time: Dict[str, List[int]] = None,
    ):
        self.candidate_generator = CandidateGenerator()
        self.behavior_adjuster = BehaviorAdjuster()
        self.preference_filter = PreferenceFilter()
        self.lifecycle_adjuster = LifecycleAdjuster()
        self.ranker = Ranker()

        self.product_department_map = product_department_map
        self.user_context_loader = user_context_loader

        self.popular_items_global = popular_items_global or []
        self.popular_items_by_lifecycle = popular_items_by_lifecycle or {}
        self.popular_items_by_behavior = popular_items_by_behavior or {}
        self.popular_items_by_time = popular_items_by_time or {}

        logger.info("HybridRecommender initialized (SOURCE-SAFE)")

    # ==========================================================
    # Public API
    # ==========================================================
    def recommend(
        self,
        user_id: int,
        basket: List[str],
        time_bucket: str,
        is_weekend: bool,
        top_k: int = DEFAULT_TOP_K,
        return_metadata: bool = False,
    ):
        """
        Return:
        - default: List[item_id]
        - return_metadata=True:
            List[{item_id, score, source}], metadata
        """

        metadata = {
            "rule_candidates": 0,
            "fallback_used": False,
            "insurance_used": False,
            "final_returned": 0,
        }

        # ------------------------------
        # 0. User context
        # ------------------------------
        user_context = self._build_user_context(
            user_id=user_id,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
        )

        # ------------------------------
        # 1. Rule-based recall
        # ------------------------------
        candidates, rule_scores, rule_sources = self.candidate_generator.generate(
            basket=basket,
            user_context=user_context,
            top_k=top_k * 3,
        )

        metadata["rule_candidates"] = len(candidates)

        for pid in candidates:
            rule_sources.setdefault(pid, set()).add(RULE)

        # ------------------------------
        # 1.5 Fallback recall (if needed)
        # ------------------------------
        if len(candidates) < top_k:
            need = top_k - len(candidates)

            logger.warning(
                f"user_id={user_id} | rule_candidates={len(candidates)} | fallback_needed={need}"
            )

            fb_items, fb_scores, fb_sources = self._fallback_recall(
                basket=basket,
                user_context=user_context,
                top_k=need * 3,
            )

            metadata["fallback_used"] = True

            for pid in fb_items:
                if pid in rule_sources:
                    continue

                rule_sources[pid] = fb_sources.get(pid, set())
                candidates.append(pid)

        if not candidates:
            logger.error(f"user_id={user_id} | empty recall set")
            if return_metadata:
                return [], metadata
            return []

        # ------------------------------
        # 2. Behavior adjustment
        # ------------------------------
        behavior_scores = self.behavior_adjuster.apply(
            scores=rule_scores,
            behavior_cluster=int(user_context["behavior_cluster"]),
        )

        # ------------------------------
        # 3. Preference scoring
        # ------------------------------
        preference_scores = self.preference_filter.apply(
            candidates=candidates,
            preference_cluster=int(user_context["preference_cluster"]),
            product_department_map=self.product_department_map,
        )

        # ------------------------------
        # 4. Lifecycle adjustment
        # ------------------------------
        lifecycle_scores = self.lifecycle_adjuster.adjust(
            scores=preference_scores,
            lifecycle_stage=user_context["lifecycle_stage"],
        )

        # ======================================================
        # 5. BUILD ITEM POOL (SOURCE-SAFE)
        # ======================================================
        item_pool = []
        for pid in candidates:
            item_pool.append({
                "item_id": pid,
                "rule_score": rule_scores.get(pid, 0.0),
                "behavior_score": behavior_scores.get(pid, 0.0),
                "preference_score": preference_scores.get(pid, 0.0),
                "lifecycle_score": lifecycle_scores.get(pid, 0.0),
                "source": set(rule_sources.get(pid, [])),
            })

        # ------------------------------
        # 6. Ranking (NO SOURCE LOSS)
        # ------------------------------
        ranked_items = self.ranker.rank(
            rule_scores=rule_scores,
            behavior_scores=behavior_scores,
            preference_scores=preference_scores,
            lifecycle_scores=lifecycle_scores,
            top_k=top_k,
            return_scores=True,
        )


        final_results = []

        for item_id, score in ranked_items:
            # Separate context levels (L1-L4) from source types (RULE, POPULAR, etc.)
            item_sources = rule_sources.get(item_id, {INSURANCE})
            context_levels = sorted([s for s in item_sources if s.startswith("L")])
            source_types = sorted([s for s in item_sources if not s.startswith("L")])
            
            # If no source type found, mark as RULE (from context matching)
            if not source_types and context_levels:
                source_types = [RULE]
            
            final_results.append({
                "item_id": item_id,
                "score": score,
                "source": source_types if source_types else [INSURANCE],
                "context_level": context_levels,  # L1, L2, L3, L4, or []
            })



        # ------------------------------
        # 7. Insurance fill (guarantee top_k)
        # ------------------------------
        if len(final_results) < top_k:
            need = top_k - len(final_results)

            logger.warning(
                f"user_id={user_id} | rank_returned={len(final_results)} | insurance_fill={need}"
            )

            metadata["insurance_used"] = True
            existing = {r["item_id"] for r in final_results}

            for pid in self.popular_items_global:
                if pid not in existing:
                    final_results.append({
                        "item_id": pid,
                        "score": 0.0,
                        "source": [INSURANCE],
                    })
                if len(final_results) >= top_k:
                    break

        metadata["final_returned"] = len(final_results)

        # ------------------------------
        # Source stats (debug / eval)
        # ------------------------------
        metadata.update({
            "rule_item_count": sum(RULE in r["source"] for r in final_results),
            "popular_item_count": sum(POPULAR in r["source"] for r in final_results),
            "insurance_item_count": sum(INSURANCE in r["source"] for r in final_results),
        })

        logger.info(
            f"user_id={user_id} | basket={len(basket)} | returned={len(final_results)}"
        )

        if return_metadata:
            return final_results, metadata

        return [r["item_id"] for r in final_results]

    # ==========================================================
    # Fallback recall
    # ==========================================================
    def _fallback_recall(
        self,
        basket: List[str],
        user_context: Dict[str, Any],
        top_k: int,
    ) -> Tuple[List[int], Dict[int, float], Dict[int, set]]:

        scores = {}
        sources = defaultdict(set)

        lifecycle = user_context["lifecycle_stage"]
        behavior = int(user_context["behavior_cluster"])

        time_bucket = user_context["time_bucket"]
        
        popular = (
            self.popular_items_by_time.get(time_bucket)
            or self.popular_items_by_lifecycle.get(lifecycle)
            or self.popular_items_by_behavior.get(behavior)
            or self.popular_items_global
        )

        if popular:
            items = popular[:top_k]
            for i, pid in enumerate(items):
                scores[pid] = 1.0 - i * 0.001
                sources[pid].add(POPULAR)
            return items, scores, sources

        basket_depts = {
            self.product_department_map.get(int(pid))
            for pid in basket
            if int(pid) in self.product_department_map
        }

        similar_items = [
            pid
            for pid, dept in self.product_department_map.items()
            if dept in basket_depts
        ]

        if similar_items:
            items = similar_items[:top_k]
            for i, pid in enumerate(items):
                scores[pid] = 0.8 - i * 0.001
                sources[pid].add(SIMILAR_DEPT)
            return items, scores, sources

        items = self.popular_items_global[:top_k]
        for i, pid in enumerate(items):
            scores[pid] = 0.5 - i * 0.001
            sources[pid].add(INSURANCE)

        return items, scores, sources

    # ==========================================================
    # User context
    # ==========================================================
    def _build_user_context(
        self,
        user_id: int,
        time_bucket: str,
        is_weekend: bool,
    ) -> Dict[str, Any]:

        clusters = self.user_context_loader.get_user_context(user_id)

        return {
            "user_id": user_id,
            "time_bucket": time_bucket,
            "is_weekend": is_weekend,
            "behavior_cluster": clusters["behavior_cluster"],
            "preference_cluster": clusters["preference_cluster"],
            "lifecycle_stage": clusters["lifecycle_stage"],
        }
