import logging
from typing import List

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    lit,
    size,
    expr,
    trim,
    row_number,
    rand,
    concat_ws,
)

from src.config.settings import (
    CONTEXT_HIERARCHY,
    CONTEXT_MIN_TXNS_BY_LEVEL,
    FPGROWTH_MAX_ITEMS_PER_TXN,
    SPARK_SHUFFLE_PARTITIONS,
)

logger = logging.getLogger(__name__)

# =================================================
# SAFETY GUARDS
# =================================================
MAX_TXNS_PER_CONTEXT = 300_000
MIN_ITEMS_PER_TXN = 2


class ContextTransactionBuilder:
    """
    Build context-aware transactions for FP-Growth.

    INPUT columns:
        - products : string "[1,2,3]"
        - all context dimensions used in CONTEXT_HIERARCHY

    OUTPUT columns:
        - context_level : L1..L5
        - context_key   : string
        - items         : array<string>
    """

    # =================================================
    # LOG SAMPLE CONTEXTS (FOR DEBUG / INSPECTION)
    # =================================================
    def _log_context_samples(
        self,
        level: str,
        df: DataFrame,
        n: int = 3,
    ) -> None:
        """
        Log sample context keys in format:
        Lx::key=value|key=value
        """
        try:
            samples = (
                df
                .select("context_key")
                .distinct()
                .limit(n)
                .collect()
            )

            for row in samples:
                logger.info(f"{level}::{row['context_key']}")
        except Exception as e:
            logger.warning(
                f"[{level}] Failed to log context samples: {e}"
            )

    # =================================================
    # Build ONE level
    # =================================================
    def _build_level(
        self,
        df: DataFrame,
        level: str,
        dims: List[str],
        min_txns: int,
    ) -> DataFrame:

        logger.info(
            f"Building {level} | "
            f"dims={dims if dims else 'GLOBAL'} | "
            f"min_txns={min_txns}"
        )

        # -------------------------------------------------
        # Build context_key
        # -------------------------------------------------
        if dims:
            ctx_exprs = [expr(f"concat('{d}=', {d})") for d in dims]
            df_level = df.withColumn(
                "context_key",
                concat_ws("|", *ctx_exprs)
            )
        else:
            df_level = df.withColumn(
                "context_key",
                lit("GLOBAL")
            )

        level_df = (
            df_level
            .withColumn("context_level", lit(level))
            .select("context_level", "context_key", "items")
            .where(size(col("items")) >= MIN_ITEMS_PER_TXN)
        )

        # -------------------------------------------------
        # Filter sparse contexts
        # -------------------------------------------------
        if min_txns > 0:
            valid_ctx = (
                level_df
                .groupBy("context_key")
                .count()
                .where(col("count") >= min_txns)
                .select("context_key")
            )

            level_df = level_df.join(
                valid_ctx,
                on="context_key",
                how="inner",
            )

        # -------------------------------------------------
        # Cap transactions per context
        # -------------------------------------------------
        w = Window.partitionBy("context_key").orderBy(rand())

        level_df = (
            level_df
            .withColumn("rn", row_number().over(w))
            .where(col("rn") <= MAX_TXNS_PER_CONTEXT)
            .drop("rn")
        )

        return level_df.repartition(
            SPARK_SHUFFLE_PARTITIONS,
            col("context_key")
        )

    # =================================================
    # Public API
    # =================================================
    def build(self, df: DataFrame) -> DataFrame:

        logger.info("=" * 80)
        logger.info("START CONTEXT TRANSACTION BUILDING")
        logger.info("=" * 80)

        # -------------------------------------------------
        # Validate required columns
        # -------------------------------------------------
        required_cols = {"products"}
        for dims in CONTEXT_HIERARCHY.values():
            required_cols.update(dims)

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # -------------------------------------------------
        # Parse products → items
        # -------------------------------------------------
        df = (
            df
            .withColumn(
                "products_clean",
                expr("regexp_replace(products, '[\\\\[\\\\]]', '')")
            )
            .withColumn(
                "items",
                expr(
                    "filter("
                    "transform(split(products_clean, ','), x -> trim(x)), "
                    "x -> x is not null and x != ''"
                    ")"
                )
            )
            .drop("products", "products_clean")
        )

        # -------------------------------------------------
        # Limit items per transaction
        # -------------------------------------------------
        if FPGROWTH_MAX_ITEMS_PER_TXN:
            df = df.withColumn(
                "items",
                expr(
                    f"slice(items, 1, {FPGROWTH_MAX_ITEMS_PER_TXN})"
                )
            )

        # -------------------------------------------------
        # Cache base df
        # -------------------------------------------------
        df = df.cache()
        base_cnt = df.count()
        logger.info(f"Base transactions cached: {base_cnt:,}")

        level_dfs: List[DataFrame] = []

        # -------------------------------------------------
        # Build each level
        # -------------------------------------------------
        for level, dims in CONTEXT_HIERARCHY.items():
            min_txns = CONTEXT_MIN_TXNS_BY_LEVEL.get(level, 0)

            level_df = self._build_level(
                df=df,
                level=level,
                dims=dims,
                min_txns=min_txns,
            )

            # 👇 LOG SAMPLE CONTEXTS (L1::..., L2::..., ...)
            self._log_context_samples(
                level=level,
                df=level_df,
                n=3,
            )

            level_dfs.append(level_df)

        # -------------------------------------------------
        # Union all levels
        # -------------------------------------------------
        result_df = level_dfs[0]
        for part in level_dfs[1:]:
            result_df = result_df.unionByName(part)

        df.unpersist()

        logger.info(
            "Context transaction building completed | "
            f"levels={list(CONTEXT_HIERARCHY.keys())}"
        )

        return result_df
