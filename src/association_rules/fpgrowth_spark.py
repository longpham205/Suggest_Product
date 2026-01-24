# src/association_rules/fpgrowth_spark.py

"""
Spark FP-Growth Miner (Context-aware, Production-safe)

Responsibilities:
- Mine association rules per context
- Enforce hard safety caps to avoid OOM
- Log mining statistics (CSV + JSONL)
- Return compact Python rule index for downstream RuleBuilder

Design notes:
- Spark only used for mining
- All heavy safety checks are done BEFORE collect()
"""

import os
import sys
import time
import logging
from typing import Dict, List
from collections import defaultdict

# -------------------------------------------------
# Path setup
# -------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import col, size, count, desc

from src.config.settings import (
    FPGROWTH_MIN_SUPPORT,
    FPGROWTH_MIN_CONFIDENCE,
    FPGROWTH_MIN_LIFT,
    FPGROWTH_MAX_ANTECEDENT_LEN,
    FPGROWTH_MAX_RULES_PER_ANTECEDENT,
    MAX_RULES_PER_CONTEXT,
    RULE_SCORE_CONF_WEIGHT,
    RULE_SCORE_LIFT_WEIGHT,
)

from src.association_rules.miner_stats import save_miner_stats

logger = logging.getLogger(__name__)


class SparkFPGrowthMiner:
    """
    Context-aware FP-Growth miner using Spark MLlib.
    Designed for large-scale offline mining.
    """

    # =================================================
    # INIT
    # =================================================
    def __init__(
        self,
        spark: SparkSession,
        min_support: float = FPGROWTH_MIN_SUPPORT,
        min_confidence: float = FPGROWTH_MIN_CONFIDENCE,
        min_lift: float = FPGROWTH_MIN_LIFT,
        max_antecedent_len: int = FPGROWTH_MAX_ANTECEDENT_LEN,
    ):
        self.spark = spark
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.max_antecedent_len = int(max_antecedent_len)

        logger.info(
            "SparkFPGrowthMiner initialized | "
            f"support={self.min_support}, "
            f"confidence={self.min_confidence}, "
            f"lift={self.min_lift}, "
            f"max_ant_len={self.max_antecedent_len}"
        )

    # =================================================
    # Validation
    # =================================================
    @staticmethod
    def _validate_input(df: DataFrame) -> None:
        if "items" not in df.columns:
            raise ValueError("Input DataFrame must contain column: items")

    # =================================================
    # Rule scoring
    # =================================================
    @staticmethod
    def _score_rule(confidence: float, lift: float) -> float:
        lift = min(lift, 10.0)
        return (
            RULE_SCORE_CONF_WEIGHT * confidence
            + RULE_SCORE_LIFT_WEIGHT * lift
        )

    # =================================================
    # Adaptive safety parameters
    # =================================================
    def _adaptive_params(self, n_txns: int):
        min_support = self.min_support
        max_ant_len = self.max_antecedent_len

        if n_txns > 500_000:
            min_support = max(min_support, 0.02)
            max_ant_len = min(max_ant_len, 1)
        elif n_txns > 200_000:
            min_support = max(min_support, 0.01)
            max_ant_len = min(max_ant_len, 2)

        return min_support, max_ant_len

    # =================================================
    # Mine ONE context
    # =================================================
    def mine_single_context(
        self,
        df: DataFrame,
        context_key: str,
    ) -> Dict[str, List[Dict]]:

        self._validate_input(df)
        start_time = time.time()

        n_txns = df.count()
        if n_txns < 100:
            logger.info(f"[{context_key}] Skipped ({n_txns} txns)")
            return {}

        min_support, max_ant_len = self._adaptive_params(n_txns)

        logger.info(
            f"[{context_key}] Mining {n_txns:,} txns | "
            f"support={min_support}, max_ant_len={max_ant_len}"
        )

        try:
            model = FPGrowth(
                itemsCol="items",
                minSupport=min_support,
                minConfidence=self.min_confidence,
            ).fit(df)

            rules_df = (
                model.associationRules
                .filter(
                    (col("lift") >= self.min_lift)
                    & (size(col("antecedent")) <= max_ant_len)
                    & (size(col("consequent")) == 1)
                )
                .orderBy(desc("lift"), desc("confidence"))
                .limit(MAX_RULES_PER_CONTEXT)
            )

            rules = rules_df.collect()
            if not rules:
                save_miner_stats(
                    context=context_key,
                    num_transactions=n_txns,
                    min_support=min_support,
                    min_confidence=self.min_confidence,
                    runtime_seconds=time.time() - start_time,
                    status="success",
                )
                logger.info(f"[{context_key}] No rules found")
                return {}

            rule_index = self._rules_to_index_from_rows(rules)
            num_rules = sum(len(v) for v in rule_index.values())

            confidences = [
                r["confidence"]
                for rules in rule_index.values()
                for r in rules
            ]

            save_miner_stats(
                context=context_key,
                num_transactions=n_txns,
                min_support=min_support,
                min_confidence=self.min_confidence,
                num_frequent_itemsets=model.freqItemsets.count(),
                num_rules=num_rules,
                avg_confidence=sum(confidences) / max(len(confidences), 1),
                max_confidence=max(confidences, default=0.0),
                runtime_seconds=time.time() - start_time,
                status="success",
            )

            logger.info(
                f"[{context_key}] Done | "
                f"rules={num_rules} | "
                f"time={time.time() - start_time:.1f}s"
            )

            return rule_index

        except Exception as e:
            save_miner_stats(
                context=context_key,
                num_transactions=n_txns,
                min_support=min_support,
                min_confidence=self.min_confidence,
                runtime_seconds=time.time() - start_time,
                status="error",
                error=str(e),
            )

            logger.exception(f"[{context_key}] FP-Growth failed")
            return {}

    # =================================================
    # Mine ALL contexts
    # =================================================
    def mine_all_contexts_full_load(
        self,
        full_df: DataFrame,
        context_col: str,
        min_txns: int = 0,
    ) -> Dict[str, Dict[str, List[Dict]]]:

        self._validate_input(full_df)
        full_df = full_df.cache()

        ctx_counts = (
            full_df.groupBy(context_col)
            .agg(count("*").alias("n"))
            .collect()
        )

        results: Dict[str, Dict[str, List[Dict]]] = {}

        for i, row in enumerate(ctx_counts, start=1):
            ctx = row[context_col]
            n = row["n"]

            if n < min_txns:
                continue

            logger.info(f"[{i}/{len(ctx_counts)}] {ctx} ({n:,} txns)")

            ctx_df = (
                full_df
                .filter(col(context_col) == ctx)
                .select("items")
            )

            rules = self.mine_single_context(ctx_df, ctx)
            if rules:
                results[ctx] = rules

        full_df.unpersist()

        logger.info(
            f"FP-Growth completed | contexts_with_rules={len(results):,}"
        )

        return results

    # =================================================
    # Convert Spark rules → Python index
    # =================================================
    def _rules_to_index_from_rows(
        self,
        rules: List,
    ) -> Dict[str, List[Dict]]:

        rule_index: Dict[str, List[Dict]] = defaultdict(list)

        for row in rules:
            antecedent = tuple(sorted(row.antecedent))
            consequent = row.consequent[0]

            confidence = float(row.confidence)
            lift = float(row.lift)
            score = self._score_rule(confidence, lift)

            ant_key = "|".join(antecedent)

            rule_index[ant_key].append({
                "consequent": int(consequent),
                "confidence": round(confidence, 4),
                "lift": round(lift, 4),
                "score": round(score, 4),
            })

        for ant in rule_index:
            rule_index[ant] = sorted(
                rule_index[ant],
                key=lambda x: x["score"],
                reverse=True,
            )[:FPGROWTH_MAX_RULES_PER_ANTECEDENT]

        return dict(rule_index)
