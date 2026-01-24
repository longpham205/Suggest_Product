# main/run_eval.py
"""
Main entry for Hybrid Recommendation System

1. Single-user / few-user qualitative test
2. Offline evaluation (precision, recall, HR, coverage, source stats)
"""

import os
import sys
import random
import pandas as pd

# ======================================================
# PATH SETUP
# ======================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.config.settings import (
    PRODUCTS_PATH,
    AISLES_PATH,
    DEPARTMENTS_PATH,
    ORDERS_PATH,
    ORDER_PRIOR_PATH,
    OFFLINE_EVALUATION_DIR,
    DEFAULT_TOP_K,
)

from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.user_context_loader import UserContextLoader
from src.evaluation.offline_eval import OfflineEvaluator


# ======================================================
# 1. PRODUCT → DEPARTMENT MAP
# ======================================================
def build_product_department_map():
    """
    product_id (int) -> department (str)
    Used for similarity fallback
    """
    products = pd.read_csv(PRODUCTS_PATH)
    aisles = pd.read_csv(AISLES_PATH)
    departments = pd.read_csv(DEPARTMENTS_PATH)

    for df in (products, aisles, departments):
        df.columns = df.columns.str.strip()

    products = products.merge(aisles, on="aisle_id", how="left")
    products = products.merge(departments, on="department_id", how="left")

    products["department"] = products["department"].fillna("unknown")
    products["product_id"] = products["product_id"].astype(int)

    return dict(zip(products["product_id"], products["department"]))


# ======================================================
# POPULAR ITEMS (GLOBAL)
# ======================================================
def build_popular_items_global(top_n: int = 200) -> list[int]:
    """
    Global popular products by frequency
    Used for fallback & insurance recall
    """
    prior = pd.read_csv(ORDER_PRIOR_PATH)
    prior.columns = prior.columns.str.strip()

    popular = (
        prior["product_id"]
        .value_counts()
        .head(top_n)
        .index
        .astype(int)
        .tolist()
    )

    return popular


# ======================================================
# 2. SINGLE / FEW USER TEST (DEBUG)
# ======================================================
def run_single_user_test(
    recommender: HybridRecommender,
    user_context_loader: UserContextLoader,
    max_users: int = 5,
    basket_size: int = 3,
):
    print("\n==============================")
    print(" SINGLE USER TEST")
    print("==============================")

    orders = pd.read_csv(ORDERS_PATH)
    prior = pd.read_csv(ORDER_PRIOR_PATH)

    orders.columns = orders.columns.str.strip()
    prior.columns = prior.columns.str.strip()

    prior = prior.merge(
        orders[["order_id", "user_id"]],
        on="order_id",
        how="left",
    )

    user_histories = (
        prior.groupby("user_id")["product_id"]
        .apply(lambda x: list(map(int, x)))
        .reset_index()
    )

    user_histories = user_histories[
        user_histories["product_id"].apply(len) >= basket_size + 1
    ].sample(n=min(max_users, len(user_histories)), random_state=42)

    for _, row in user_histories.iterrows():
        user_id = int(row["user_id"])
        history = row["product_id"]

        basket = random.sample(history, basket_size)

        time_bucket = "morning"
        is_weekend = False

        profile = user_context_loader.get_user_context(user_id)

        print(f"\nUser {user_id}")
        print(f"Basket: {basket}")
        print(
            f"Context: "
            f"behavior={profile['behavior_cluster']}, "
            f"preference={profile['preference_cluster']}, "
            f"lifecycle={profile['lifecycle_stage']}"
        )

        recs = recommender.recommend(
            user_id=user_id,
            basket=basket,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
            top_k=DEFAULT_TOP_K,
        )

        if not recs:
            print("→ No recommendation")
            continue

        print("Recommendations:")
        for i, pid in enumerate(recs, 1):
            print(f"{i:02d}. Product {pid}")


# ======================================================
# 3. OFFLINE EVALUATION
# ======================================================
def run_offline_evaluation(
    recommender: HybridRecommender,
    max_users: int = 1000,
    k: int = 10,
):
    import time

    print("\n==============================")
    print(" OFFLINE EVALUATION")
    print("==============================")
    print(f"Max users: {max_users}, K: {k}")

    evaluator = OfflineEvaluator(
        recommender=recommender,
        max_users=max_users,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OFFLINE_EVALUATION_DIR, exist_ok=True)
    save_path = f"{OFFLINE_EVALUATION_DIR}/eval_k{k}_{ts}.json"

    results = evaluator.evaluate(k=k, save_path=save_path)

    print("\nEvaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:30s}: {value:.4f}")
        else:
            print(f"{metric:30s}: {value}")

    print(f"\nSaved to: {save_path}")


# ======================================================
# 4. MAIN
# ======================================================
def main():
    print("\n===================================")
    print(" HYBRID RECOMMENDER SYSTEM")
    print("===================================")

    product_department_map = build_product_department_map()
    user_context_loader = UserContextLoader()
    popular_items_global = build_popular_items_global(top_n=200)

    recommender = HybridRecommender(
        product_department_map=product_department_map,
        user_context_loader=user_context_loader,
        popular_items_global=popular_items_global,
    )
    
    # -------------------------------------------------- 
    # 1. Single-user debug 
    # --------------------------------------------------  
    # run_single_user_test( 
    #     recommender=recommender, 
    #     user_context_loader=user_context_loader,
    #     max_users=5,
    #     basket_size=3, 
    # )

    # -------------------------------------------------- 
    # 2. Offline evaluation 
    # --------------------------------------------------
    run_offline_evaluation(
        recommender=recommender,
        max_users=10,
        k=10,
    )

    print("\nDONE")


if __name__ == "__main__":
    main()
