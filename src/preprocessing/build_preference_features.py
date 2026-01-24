# src/preprocessing/build_preference_features.py

import pandas as pd


def build_preference_features(orders, order_products, products, departments):
    """
    Build user preference features based on departments
    """

    # 1. order_products -> orders (lấy user_id)
    df = order_products.merge(
        orders[["order_id", "user_id"]],
        on="order_id",
        how="left"
    )

    # 2. join products
    df = df.merge(
        products[["product_id", "department_id"]],
        on="product_id",
        how="left"
    )

    # 3. join departments
    df = df.merge(
        departments,
        on="department_id",
        how="left"
    )

    # 4. preference count
    pref = (
        df.groupby(["user_id", "department"])
        .size()
        .reset_index(name="purchase_count")
    )

    # 5. normalize preference
    pref["total"] = pref.groupby("user_id")["purchase_count"].transform("sum")
    pref["preference_score"] = pref["purchase_count"] / pref["total"]

    return pref[["user_id", "department", "preference_score"]]
