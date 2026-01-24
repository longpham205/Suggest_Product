# src/preprocessing/build_transactions_context_extended.py
"""
Build extended transactions với 8 context dimensions

Context dimensions:
1. time_bucket (night/morning/afternoon/evening)
2. is_weekend (0/1)
3. day_of_week (0-6)
4. basket_size_category (small/medium/large)
5. purchase_frequency (low/medium/high)
6. lifecycle_stage (new/regular/loyal/vip)
7. preference_cluster (0-4)
8. behavior_cluster (0-4)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import numpy as np

# =====================================================
# Path setup
# =====================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config.settings import (
    ORDERS_PATH,
    ORDER_PRIOR_PATH,
    ORDER_TRAIN_PATH,
    BEHAVIOR_FEATURES_PATH,
    BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH,
    PREFERENCE_CLUSTER_ASSIGNMENTS_PATH,
    LIFECYCLE_ASSIGNMENTS_PATH,
    TRANSACTIONS_CONTEXT_EXTENDED_PATH,
    CONTEXT_HIERARCHY,
    BASKET_SIZE_BINS,
    BASKET_SIZE_LABELS,
    PURCHASE_FREQ_QUANTILES,
)

# =====================================================
# SAFETY GUARDS (CRITICAL)
# =====================================================
MIN_BASKET_SIZE = 2          # FP-Growth useless below this
MAX_BASKET_SIZE = 20         # Prevent combinatorial explosion
MIN_PRODUCTS_PER_CONTEXT = 50  # Used later by miner
MAX_CONTEXT_CARDINALITY_WARN = 50_000

# =====================================================
# Logging
# =====================================================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# =====================================================
# Load cluster assignments
# =====================================================
def load_cluster_assignments() -> Dict[str, Optional[pd.DataFrame]]:
    assignments = {}

    def _load(path: Path, name: str):
        if path.exists():
            df = pd.read_csv(path)
            logger.info(f"Loaded {name}: {len(df):,}")
            return df
        logger.warning(f"{name} not found: {path}")
        return None

    assignments["behavior"] = _load(
        BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH, "behavior clusters"
    )
    assignments["preference"] = _load(
        PREFERENCE_CLUSTER_ASSIGNMENTS_PATH, "preference clusters"
    )
    assignments["lifecycle"] = _load(
        LIFECYCLE_ASSIGNMENTS_PATH, "lifecycle stages"
    )

    return assignments


# =====================================================
# Purchase frequency
# =====================================================
def compute_purchase_frequency(
    behavior_features: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:

    if behavior_features is None or "total_orders" not in behavior_features.columns:
        logger.warning("Behavior features missing → default purchase_frequency")
        return None

    df = behavior_features[["user_id", "total_orders"]].copy()

    q_low, q_high = df["total_orders"].quantile(
        PURCHASE_FREQ_QUANTILES
    ).values

    def categorize(x):
        if x <= q_low:
            return "low"
        elif x <= q_high:
            return "medium"
        return "high"

    df["purchase_frequency"] = df["total_orders"].apply(categorize)

    logger.info(
        f"Purchase frequency distribution:\n"
        f"{df['purchase_frequency'].value_counts()}"
    )

    return df[["user_id", "purchase_frequency"]]


# =====================================================
# Context key builder
# =====================================================
def build_context_key(row: pd.Series, dimensions: list) -> str:
    if not dimensions:
        return "global"

    parts = []
    for dim in dimensions:
        val = row.get(dim, "unknown")
        if pd.isna(val):
            val = "unknown"
        parts.append(f"{dim}={val}")

    return "|".join(parts)


# =====================================================
# MAIN PIPELINE
# =====================================================
def build_transactions_context_extended(
    save_parquet: bool = True,
    sample_ratio: Optional[float] = None,
) -> pd.DataFrame:

    logger.info("=" * 70)
    logger.info("BUILD EXTENDED CONTEXT TRANSACTIONS (SAFE MODE)")
    logger.info("=" * 70)

    # =================================================
    # 1. Load orders
    # =================================================
    orders = pd.read_csv(ORDERS_PATH)

    if sample_ratio and 0 < sample_ratio < 1:
        orders = orders.sample(frac=sample_ratio, random_state=42)
        logger.info(f"Sampled orders: {len(orders):,}")

    logger.info(f"Orders loaded: {len(orders):,}")

    # =================================================
    # 2. Load order products
    # =================================================
    order_products = pd.concat(
        [
            pd.read_csv(ORDER_PRIOR_PATH),
            pd.read_csv(ORDER_TRAIN_PATH),
        ],
        ignore_index=True,
    )

    logger.info(f"Order-product rows: {len(order_products):,}")

    # =================================================
    # 3. Aggregate products per order
    # =================================================
    products_per_order = (
        order_products
        .groupby("order_id")["product_id"]
        .apply(list)
        .reset_index(name="products")
    )

    # =================================================
    # 4. Merge orders + products
    # =================================================
    df = orders.merge(products_per_order, on="order_id", how="inner")
    logger.info(f"Orders with products: {len(df):,}")

    # =================================================
    # 5. BASKET SAFETY GUARDS (CRITICAL)
    # =================================================
    df["products"] = df["products"].apply(
        lambda x: x[:MAX_BASKET_SIZE] if len(x) > MAX_BASKET_SIZE else x
    )
    df["basket_size"] = df["products"].apply(len)

    before = len(df)
    df = df[df["basket_size"] >= MIN_BASKET_SIZE]
    logger.info(
        f"Filtered small baskets (<{MIN_BASKET_SIZE}): "
        f"{before - len(df):,} removed"
    )

    # =================================================
    # 6. TEMPORAL CONTEXT
    # =================================================
    df["time_bucket"] = pd.cut(
        df["order_hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
        right=False,
    ).astype(str)

    df["is_weekend"] = df["order_dow"].isin([0, 6]).astype(int)
    df["day_of_week"] = df["order_dow"].astype(int)

    df["basket_size_category"] = pd.cut(
        df["basket_size"],
        bins=BASKET_SIZE_BINS,
        labels=BASKET_SIZE_LABELS,
        right=False,
    ).astype(str).fillna("medium")

    # =================================================
    # 7. USER SEGMENTS
    # =================================================
    assignments = load_cluster_assignments()

    # Purchase frequency
    if BEHAVIOR_FEATURES_PATH.exists():
        behavior_features = pd.read_csv(BEHAVIOR_FEATURES_PATH)
        purchase_freq = compute_purchase_frequency(behavior_features)
        if purchase_freq is not None:
            df = df.merge(purchase_freq, on="user_id", how="left")

    df["purchase_frequency"] = df["purchase_frequency"].fillna("medium")

    # Lifecycle
    if assignments["lifecycle"] is not None:
        df = df.merge(
            assignments["lifecycle"][["user_id", "lifecycle_stage"]],
            on="user_id",
            how="left",
        )

    df["lifecycle_stage"] = df["lifecycle_stage"].fillna("new")

    # Preference cluster
    if assignments["preference"] is not None:
        pref = assignments["preference"].rename(
            columns={"cluster": "preference_cluster"}
        )
        df = df.merge(pref[["user_id", "preference_cluster"]], on="user_id", how="left")

    df["preference_cluster"] = df["preference_cluster"].fillna(0).astype(int)

    # Behavior cluster
    if assignments["behavior"] is not None:
        beh = assignments["behavior"].rename(
            columns={"cluster": "behavior_cluster"}
        )
        df = df.merge(beh[["user_id", "behavior_cluster"]], on="user_id", how="left")

    df["behavior_cluster"] = df["behavior_cluster"].fillna(0).astype(int)

    # =================================================
    # 8. BUILD CONTEXT KEYS
    # =================================================
    logger.info("Building context hierarchy keys...")

    for level, dims in CONTEXT_HIERARCHY.items():
        col = f"context_{level}"
        df[col] = df.apply(lambda r: build_context_key(r, dims), axis=1)

        n_ctx = df[col].nunique()
        logger.info(f"  {level}: {n_ctx:,} unique contexts")

        if n_ctx > MAX_CONTEXT_CARDINALITY_WARN:
            logger.warning(
                f"⚠ {level} context cardinality too high: {n_ctx:,}"
            )

    # =================================================
    # 9. FINAL COLUMNS
    # =================================================
    final_cols = [
        "order_id",
        "user_id",
        "products",
        "time_bucket",
        "is_weekend",
        "day_of_week",
        "basket_size_category",
        "purchase_frequency",
        "lifecycle_stage",
        "preference_cluster",
        "behavior_cluster",
        "context_L1",
        "context_L2",
        "context_L3",
        "context_L4",
        "context_L5",
    ]

    df = df[final_cols]

    # =================================================
    # 10. SAVE PARQUET
    # =================================================
    if save_parquet:
        logger.info(f"Saving parquet → {TRANSACTIONS_CONTEXT_EXTENDED_PATH}")
        out = df.copy()
        out["products"] = out["products"].apply(str)

        TRANSACTIONS_CONTEXT_EXTENDED_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )
        out.to_parquet(
            TRANSACTIONS_CONTEXT_EXTENDED_PATH,
            index=False,
            engine="pyarrow",
        )

        logger.info(f"Saved {len(out):,} transactions")

    return df


# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None)
    args = parser.parse_args()

    df = build_transactions_context_extended(
        save_parquet=True,
        sample_ratio=args.sample,
    )

    print(df.head(3))