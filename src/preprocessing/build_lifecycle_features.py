# src/preprocessing/build_lifecycle_features.py

import pandas as pd


def build_lifecycle_features(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng đặc trưng vòng đời người dùng
    """

    lifecycle = orders.groupby("user_id").agg(
        first_order=("order_number", "min"),
        last_order=("order_number", "max"),
        total_orders=("order_id", "nunique"),
        active_days=("days_since_prior_order", "sum")
    ).reset_index()

    lifecycle["active_span"] = lifecycle["last_order"] - lifecycle["first_order"]

    # Phân nhóm lifecycle
    lifecycle["lifecycle_stage"] = pd.cut(
        lifecycle["total_orders"],
        bins=[0, 3, 10, 50, float("inf")],
        labels=["new", "regular", "loyal", "vip"]
    )

    return lifecycle
