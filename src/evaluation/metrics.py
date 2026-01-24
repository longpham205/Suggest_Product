# src/evaluation/metrics.py

from typing import List, Set, Union, Dict
from collections import Counter


# ============================================================
# Accuracy-based metrics (per-user)
# ============================================================

def precision_at_k(
    recommended: List[str],
    relevant: Union[Set[str], List[str]],
    k: int
) -> float:
    """
    Compute Precision@K.

    Precision@K = (# of recommended items in top-K that are relevant) / K
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant)
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant_set)
    return hits / k


def recall_at_k(
    recommended: List[str],
    relevant: Union[Set[str], List[str]],
    k: int
) -> float:
    """
    Compute Recall@K.

    Recall@K = (# of recommended items in top-K that are relevant)
               / (# of relevant items)
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(
    recommended: List[str],
    relevant: Union[Set[str], List[str]],
    k: int
) -> float:
    """
    Compute HitRate@K.

    HitRate@K = 1 if at least one relevant item appears in top-K,
                otherwise 0
    """
    relevant_set = set(relevant)
    recommended_k = recommended[:k]
    return 1.0 if set(recommended_k) & relevant_set else 0.0


# ============================================================
# Coverage & robustness metrics (system-level)
# ============================================================

def user_coverage(
    recommendations: Dict[int, List[str]]
) -> float:
    """
    User Coverage

    Fraction of users who receive at least one recommendation.
    """
    if not recommendations:
        return 0.0

    covered_users = sum(
        1 for recs in recommendations.values() if len(recs) > 0
    )
    return covered_users / len(recommendations)


def item_coverage(
    recommendations: Dict[int, List[str]],
    all_items: Set[str]
) -> float:
    """
    Item Coverage

    Fraction of catalog items that appear in any recommendation list.
    """
    if not all_items:
        return 0.0

    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs)

    return len(recommended_items) / len(all_items)


def rule_hit_rate(
    source_log: Dict[int, str]
) -> float:
    """
    Rule Hit Rate (Rules Coverage)

    Fraction of users for which association rules contributed
    to the final recommendation.

    Valid sources:
        - "rules"
        - "rules+fallback"
        - "fallback"
    """
    if not source_log:
        return 0.0

    rule_hits = sum(
        1 for src in source_log.values()
        if src in ("rules", "rules+fallback")
    )
    return rule_hits / len(source_log)


def source_distribution(
    source_log: Dict[int, str]
) -> Dict[str, float]:
    """
    Source Distribution

    Distribution of recommendation sources
    (rules vs fallback vs hybrid).
    """
    if not source_log:
        return {}

    counter = Counter(source_log.values())
    total = sum(counter.values())

    return {
        source: count / total
        for source, count in counter.items()
    }
