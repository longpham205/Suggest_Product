# src/association_rules/rule_builder.py

"""
RuleBuilder – Post-processing association rules (Python-only)

Responsibilities:
- Normalize rule format
- Re-score rules (config-driven)
- Deduplicate & prune weak rules
- Prepare production-ready rule index

IMPORTANT:
- This module DOES NOT depend on Spark
- Input is rule_index from SparkFPGrowthMiner
"""

import logging
from typing import Dict, List
from collections import defaultdict
import hashlib

from src.config.settings import (
    FPGROWTH_MIN_LIFT,
    FPGROWTH_MAX_RULES_PER_ANTECEDENT,
    RULE_SCORE_CONF_WEIGHT,
    RULE_SCORE_LIFT_WEIGHT,
)

logger = logging.getLogger(__name__)


class RuleBuilder:
    """
    Post-process FP-Growth rules for production inference.
    """

    def __init__(
        self,
        min_lift: float = FPGROWTH_MIN_LIFT,
        max_rules_per_antecedent: int = FPGROWTH_MAX_RULES_PER_ANTECEDENT,
    ):
        self.min_lift = min_lift
        self.max_rules_per_antecedent = max_rules_per_antecedent

        self._conf_weight = RULE_SCORE_CONF_WEIGHT
        self._lift_weight = RULE_SCORE_LIFT_WEIGHT
        self._support_weight = max(
            0.0, 1.0 - self._conf_weight - self._lift_weight
        )

        if self._support_weight == 0.0:
            logger.warning(
                "Rule score support weight is zero. "
                "Check RULE_SCORE_CONF_WEIGHT and RULE_SCORE_LIFT_WEIGHT."
            )

        logger.info(
            "RuleBuilder initialized | "
            f"min_lift={self.min_lift} | "
            f"max_rules_per_ant={self.max_rules_per_antecedent} | "
            f"weights=(conf={self._conf_weight}, "
            f"lift={self._lift_weight}, "
            f"support={self._support_weight})"
        )

    # =================================================
    # Public API
    # =================================================
    def build(
        self,
        rule_index: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        """
        Clean & optimize rule index.
        """

        logger.info("RuleBuilder | Start post-processing")

        cleaned_index: Dict[str, List[Dict]] = {}

        stats = {
            "rules_in": 0,
            "rules_invalid": 0,
            "rules_low_lift": 0,
            "rules_kept": 0,
            "antecedents_in": len(rule_index),
            "antecedents_out": 0,
        }

        for antecedent_key, rules in rule_index.items():
            if not rules:
                continue

            antecedent_items = self._parse_antecedent(antecedent_key)

            stats["rules_in"] += len(rules)

            best_by_consequent: Dict[int, Dict] = {}

            for r in rules:
                if not self._is_valid_rule(r):
                    stats["rules_invalid"] += 1
                    continue

                if float(r["lift"]) < self.min_lift:
                    stats["rules_low_lift"] += 1
                    continue

                candidate = self._normalize_rule(
                    rule=r,
                    antecedent_items=antecedent_items,
                )

                prev = best_by_consequent.get(candidate["consequent"])
                if prev is None or candidate["score"] > prev["score"]:
                    best_by_consequent[candidate["consequent"]] = candidate

            if not best_by_consequent:
                continue

            sorted_rules = sorted(
                best_by_consequent.values(),
                key=lambda x: x["score"],
                reverse=True,
            )[: self.max_rules_per_antecedent]

            cleaned_index[antecedent_key] = sorted_rules
            stats["rules_kept"] += len(sorted_rules)
            stats["antecedents_out"] += 1

        logger.info(
            "RuleBuilder completed | "
            f"Antecedents {stats['antecedents_in']} → {stats['antecedents_out']} | "
            f"Rules {stats['rules_in']} → {stats['rules_kept']} | "
            f"Dropped invalid={stats['rules_invalid']}, "
            f"low_lift={stats['rules_low_lift']}"
        )

        return cleaned_index

    # =================================================
    # Validation
    # =================================================
    @staticmethod
    def _is_valid_rule(rule: Dict) -> bool:
        return {"consequent", "confidence", "lift"}.issubset(rule.keys())

    # =================================================
    # Antecedent handling
    # =================================================
    @staticmethod
    def _parse_antecedent(antecedent_key: str) -> List[str]:
        """
        Ensure antecedent is normalized & sorted
        """
        return sorted(antecedent_key.split("|"))

    # =================================================
    # Normalization & scoring
    # =================================================
    def _normalize_rule(
        self,
        rule: Dict,
        antecedent_items: List[str],
    ) -> Dict:
        """
        Normalize rule fields and compute unified score.
        """

        confidence = float(rule["confidence"])
        lift = min(float(rule["lift"]), 10.0)
        support = float(rule.get("support", 0.0))

        score = self._score(confidence, lift, support)

        rule_id = self._make_rule_id(
            antecedent_items,
            rule["consequent"],
        )

        return {
            "rule_id": rule_id,
            "antecedent": antecedent_items,
            "consequent": int(rule["consequent"]),
            "confidence": round(confidence, 4),
            "lift": round(lift, 4),
            "support": round(support, 6),
            "score": score,
        }

    def _score(self, confidence: float, lift: float, support: float) -> float:
        score = (
            confidence * self._conf_weight
            + lift * self._lift_weight
            + support * self._support_weight
        )
        return round(score, 4)

    # =================================================
    # Rule identity
    # =================================================
    @staticmethod
    def _make_rule_id(
        antecedent_items: List[str],
        consequent: int,
    ) -> str:
        """
        Stable rule hash for tracking & debugging
        """
        raw = "|".join(antecedent_items) + "->" + str(consequent)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
