# src/preprocessing/clean_data.py
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def clean_instacart_data(data: dict) -> dict:
    """
    data = {
        orders, order_products, products, aisles, departments
    }
    """

    orders = data["orders"].copy()
    order_products = data["order_products"].copy()
    products = data["products"].copy()

    # ---- ORDERS ----
    orders.dropna(subset=["user_id"], inplace=True)
    orders["order_dow"] = orders["order_dow"].astype("int8")
    orders["order_hour_of_day"] = orders["order_hour_of_day"].astype("int8")

    # ---- ORDER_PRODUCTS ----
    order_products.drop_duplicates(
        subset=["order_id", "product_id"], inplace=True
    )
    order_products["reordered"] = order_products["reordered"].astype("int8")

    # ---- PRODUCTS ----
    products["product_name"] = (
        products["product_name"]
        .str.lower()
        .str.strip()
    )

    logger.info("Cleaned orders, order_products, products")

    return {
        "orders": orders,
        "order_products": order_products,
        "products": products,
        "aisles": data["aisles"],
        "departments": data["departments"]
    }
