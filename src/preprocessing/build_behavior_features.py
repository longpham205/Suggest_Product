# src/preprocessing/build_behavior_features.py
import pandas as pd

def build_behavior_features(orders: pd.DataFrame,
                            order_products: pd.DataFrame) -> pd.DataFrame:
    """
    Build user behavior features:
    - total_orders
    - total_products
    - reorder_ratio
    - avg_days_between_orders
    """

    # 1️⃣ Join để có user_id
    op = order_products.merge(
        orders[["order_id", "user_id"]],
        on="order_id",
        how="left"
    )

    # 2️⃣ Tổng số đơn
    user_orders = orders.groupby("user_id").agg(
        total_orders=("order_id", "nunique"),
        avg_days_between_orders=("days_since_prior_order", "mean")
    )

    # 3️⃣ Hành vi mua sản phẩm
    user_products = op.groupby("user_id").agg(
        total_products=("product_id", "count"),
        reorder_ratio=("reordered", "mean")
    )

    # 4️⃣ Merge feature
    user_features = user_orders.merge(
        user_products, on="user_id", how="left"
    ).fillna(0)

    return user_features.reset_index()
