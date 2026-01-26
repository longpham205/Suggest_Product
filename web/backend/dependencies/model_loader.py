# web/backend/dependencies/model_loader.py

import logging
from functools import lru_cache
from typing import Dict, List

from src.recommendation.hybrid_recommender import HybridRecommender
from src.recommendation.user_context_loader import UserContextLoader
from src.config.settings import (
    PRODUCT_DEPARTMENT_PATH,
    POPULAR_ITEMS_GLOBAL_PATH,
    POPULAR_ITEMS_BY_LIFECYCLE_PATH,
    POPULAR_ITEMS_BY_BEHAVIOR_PATH,
)

from src.utils.io import load_json, load_pickle

logger = logging.getLogger(__name__)


# ==========================================================
# Low-level loaders
# ==========================================================
def _load_product_department_map() -> Dict[int, str]:
    """
    product_id -> department
    """
    logger.info("Loading product_department_map")
    data = load_json(PRODUCT_DEPARTMENT_PATH)
    return {int(k): v for k, v in data.items()}


def _load_popular_items_global() -> List[int]:
    logger.info("Loading popular_items_global")
    return load_pickle(POPULAR_ITEMS_GLOBAL_PATH)


def _load_popular_items_by_lifecycle() -> Dict[str, List[int]]:
    logger.info("Loading popular_items_by_lifecycle")
    return load_pickle(POPULAR_ITEMS_BY_LIFECYCLE_PATH)


def _load_popular_items_by_behavior() -> Dict[int, List[int]]:
    logger.info("Loading popular_items_by_behavior")
    return load_pickle(POPULAR_ITEMS_BY_BEHAVIOR_PATH)


def _load_popular_items_by_time() -> Dict[str, List[int]]:
    """
    Compute popular items by time bucket on startup (Optimization)
    """
    import pandas as pd
    from src.config.settings import PURCHASE_HISTORY_CSV_PATH

    logger.info("Computing popular_items_by_time from user history...")
    try:
        df = pd.read_csv(PURCHASE_HISTORY_CSV_PATH, usecols=["product_id", "order_hour_of_day"])
        
        # Define buckets consistent with training
        df["time_bucket"] = pd.cut(
            df["order_hour_of_day"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            right=False
        ).astype(str)

        popular_by_time = {}
        for bucket in ["night", "morning", "afternoon", "evening"]:
            # Top 50 items per bucket
            top_items = (
                df[df["time_bucket"] == bucket]["product_id"]
                .value_counts()
                .head(50)
                .index
                .tolist()
            )
            popular_by_time[bucket] = [int(pid) for pid in top_items]
            
        logger.info(f"Computed time-based popularity for {list(popular_by_time.keys())}")
        return popular_by_time
    except Exception as e:
        logger.warning(f"Could not compute popular_items_by_time: {e}")
        return {}


def _load_user_context_loader() -> UserContextLoader:
    logger.info("Initializing UserContextLoader")
    return UserContextLoader()


# ==========================================================
# Public dependencies
# ==========================================================
@lru_cache(maxsize=1)
def get_recommender() -> HybridRecommender:
    """
    Load and cache HybridRecommender instance

    This function will be called ONCE
    and reused for all API requests.
    """

    logger.info("Loading HybridRecommender (cached)")

    product_department_map = _load_product_department_map()
    popular_items_global = _load_popular_items_global()
    popular_items_by_lifecycle = _load_popular_items_by_lifecycle()
    popular_items_by_behavior = _load_popular_items_by_behavior()
    popular_items_by_time = _load_popular_items_by_time()

    user_context_loader = _load_user_context_loader()

    recommender = HybridRecommender(
        product_department_map=product_department_map,
        user_context_loader=user_context_loader,
        popular_items_global=popular_items_global,
        popular_items_by_lifecycle=popular_items_by_lifecycle,
        popular_items_by_behavior=popular_items_by_behavior,
        popular_items_by_time=popular_items_by_time,
    )

    logger.info("HybridRecommender ready")
    return recommender


# ==========================================================
# Service dependencies
# ==========================================================
@lru_cache(maxsize=1)
def get_user_profile_service():
    """Get cached UserProfileService instance"""
    from services.user_service import UserProfileService
    logger.info("Loading UserProfileService (cached)")
    return UserProfileService()


@lru_cache(maxsize=1)
def get_cart_service():
    """Get cached CartService instance"""
    from services.cart_service import CartService
    logger.info("Loading CartService (cached)")
    return CartService(recommender=get_recommender())


@lru_cache(maxsize=1)
def get_recommend_service():
    """Get cached RecommendService instance"""
    from services.recommend_service import RecommendService
    logger.info("Loading RecommendService (cached)")
    return RecommendService(
        recommender=get_recommender(),
        user_profile_service=get_user_profile_service(),
    )


@lru_cache(maxsize=1)
def get_product_service():
    """Get cached ProductService instance"""
    from services.product_service import ProductService
    logger.info("Loading ProductService (cached)")
    return ProductService()

