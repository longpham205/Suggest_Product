# src/association_rules/train_fpgrowth_spark.py

"""
FP-Growth Training Pipeline (Context-aware, PySpark)
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from typing import Dict, Optional

from pyspark.sql import DataFrame

# =================================================
# PATH SETUP
# =================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# =================================================
# IMPORTS
# =================================================
from src.utils.spark_utils import get_spark_session, stop_spark_session
from src.association_rules.context_transaction_builder import (
    ContextTransactionBuilder,
)
from src.association_rules.fpgrowth_spark import SparkFPGrowthMiner
from src.association_rules.rule_builder import RuleBuilder
from src.association_rules.rule_store import save_context_rule_index

from src.config.settings import (
    TRANSACTIONS_CONTEXT_EXTENDED_PATH,
    FPGROWTH_RULE_INDEX_PATH,
    FPGROWTH_MIN_SUPPORT,
    FPGROWTH_MIN_CONFIDENCE,
    FPGROWTH_MIN_LIFT,
    FPGROWTH_MAX_ANTECEDENT_LEN,
    CONTEXT_MIN_TXNS_BY_LEVEL,
    CONTEXT_HIERARCHY,
    SPARK_DRIVER_MEMORY,
)

# =================================================
# LOGGING
# =================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =================================================
# STATS
# =================================================
def _summarize_context_rule_index(context_rule_index: Dict) -> Dict[str, int]:
    return {
        "contexts": len(context_rule_index),
        "antecedents": sum(len(v) for v in context_rule_index.values()),
        "rules": sum(
            len(rules)
            for rule_index in context_rule_index.values()
            for rules in rule_index.values()
        ),
    }


# =================================================
# TRAINING
# =================================================
def train_fpgrowth_spark(
    *,
    sample_ratio: Optional[float] = None,
    save_rules: bool = True,
) -> Dict:

    start_time = time.time()

    logger.info("=" * 80)
    logger.info("FP-GROWTH TRAINING PIPELINE (CONTEXT-AWARE)")
    logger.info("=" * 80)

    if not TRANSACTIONS_CONTEXT_EXTENDED_PATH.exists():
        raise FileNotFoundError(
            f"Transactions not found: {TRANSACTIONS_CONTEXT_EXTENDED_PATH}"
        )

    if sample_ratio is not None and not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1]")

    spark = get_spark_session(
        app_name="FP_Growth_Training",
        driver_memory=SPARK_DRIVER_MEMORY,
    )
    
    spark.sparkContext.setLogLevel("ERROR")

    ctx_txn_df: Optional[DataFrame] = None

    try:
        # -------------------------------------------------
        # LOAD TRANSACTIONS
        # -------------------------------------------------
        logger.info("Loading transactions from Parquet...")
        raw_df = spark.read.parquet(
            str(TRANSACTIONS_CONTEXT_EXTENDED_PATH)
        )

        if sample_ratio and sample_ratio < 1.0:
            logger.info(f"Sampling transactions (ratio={sample_ratio})")
            raw_df = raw_df.sample(fraction=sample_ratio, seed=42)

        if not raw_df.take(1):
            logger.warning("No transactions found.")
            return {}

        # -------------------------------------------------
        # BUILD CONTEXT TRANSACTIONS
        # -------------------------------------------------
        logger.info("Building context-aware transactions...")
        builder = ContextTransactionBuilder()
        ctx_txn_df = builder.build(raw_df).cache()

        # Sanity check only (NO count)
        if not ctx_txn_df.take(1):
            logger.warning("No context transactions built.")
            return {}

        # -------------------------------------------------
        # INIT MINER
        # -------------------------------------------------
        miner = SparkFPGrowthMiner(
            spark=spark,
            min_support=FPGROWTH_MIN_SUPPORT,
            min_confidence=FPGROWTH_MIN_CONFIDENCE,
            min_lift=FPGROWTH_MIN_LIFT,
            max_antecedent_len=FPGROWTH_MAX_ANTECEDENT_LEN,
        )

        # -------------------------------------------------
        # MINE PER CONTEXT LEVEL (DESIGN-DRIVEN)
        # -------------------------------------------------
        context_rule_index: Dict = {}

        for level in CONTEXT_HIERARCHY.keys():
            min_txns = CONTEXT_MIN_TXNS_BY_LEVEL.get(level, 0)

            logger.info(
                f"Mining context level={level} | min_txns={min_txns}"
            )

            level_df = (
                ctx_txn_df
                .filter(ctx_txn_df.context_level == level)
                .select("context_key", "items")
            )

            rules_by_ctx = miner.mine_all_contexts_full_load(
                full_df=level_df,
                context_col="context_key",
                min_txns=min_txns,
            )

            if rules_by_ctx:
                context_rule_index.update(rules_by_ctx)

        if not context_rule_index:
            logger.warning("No rules mined.")
            return {}

        # -------------------------------------------------
        # POST-PROCESS RULES
        # -------------------------------------------------
        logger.info("Post-processing rules...")
        rule_builder = RuleBuilder()

        for ctx in list(context_rule_index.keys()):
            context_rule_index[ctx] = rule_builder.build(
                context_rule_index[ctx]
            )
            if not context_rule_index[ctx]:
                del context_rule_index[ctx]

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------
        if save_rules:
            save_context_rule_index(
                context_rule_index=context_rule_index,
                path=FPGROWTH_RULE_INDEX_PATH,
                algorithm="fpgrowth",
                algorithm_version=2,
            )

        # -------------------------------------------------
        # STATS
        # -------------------------------------------------
        elapsed = time.time() - start_time
        stats = _summarize_context_rule_index(context_rule_index)

        logger.info("=" * 80)
        logger.info("FP-GROWTH TRAINING COMPLETE")
        logger.info(f"Time elapsed : {elapsed / 60:.2f} minutes")
        logger.info(f"Contexts     : {stats['contexts']:,}")
        logger.info(f"Antecedents  : {stats['antecedents']:,}")
        logger.info(f"Rules        : {stats['rules']:,}")
        if save_rules:
            logger.info(f"Saved to     : {FPGROWTH_RULE_INDEX_PATH}")
        logger.info("=" * 80)

        return context_rule_index

    finally:
        if ctx_txn_df is not None:
            ctx_txn_df.unpersist(blocking=False)
        stop_spark_session(spark)


# =================================================
# CLI
# =================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train context-aware FP-Growth association rules"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Optional sampling ratio (0 < r ≤ 1)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save rule index",
    )

    args = parser.parse_args()

    train_fpgrowth_spark(
        sample_ratio=args.sample,
        save_rules=not args.no_save,
    )


if __name__ == "__main__":
    main()
