# src/preprocessing/merge_data.py
import pandas as pd


def merge_instacart_data(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all Instacart datasets into a single dataframe
    Each row represents a product in an order
    """

    df = (
        order_products
        .merge(orders, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(aisles, on="aisle_id", how="left")
        .merge(departments, on="department_id", how="left")
    )

    return df