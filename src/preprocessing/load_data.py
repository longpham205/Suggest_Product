# src/preprocessing/load_data.py
import pandas as pd
from src.config import settings


def load_raw_data():
    aisles = pd.read_csv(settings.AISLES_PATH)
    departments = pd.read_csv(settings.DEPARTMENTS_PATH)
    products = pd.read_csv(settings.PRODUCTS_PATH)
    orders = pd.read_csv(settings.ORDERS_PATH)

    prior = pd.read_csv(settings.ORDER_PRIOR_PATH)
    train = pd.read_csv(settings.ORDER_TRAIN_PATH)

    # 🔥 GỘP PRIOR + TRAIN
    order_products = pd.concat([prior, train], ignore_index=True)

    return {
        "aisles": aisles,
        "departments": departments,
        "products": products,
        "orders": orders,
        "order_products": order_products
    }
