# dependencies/model_loader.py

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


def _load_user_context_loader() -> UserContextLoader:
    logger.info("Initializing UserContextLoader")
    return UserContextLoader()


# ==========================================================
# Public dependency
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

    user_context_loader = _load_user_context_loader()

    recommender = HybridRecommender(
        product_department_map=product_department_map,
        user_context_loader=user_context_loader,
        popular_items_global=popular_items_global,
        popular_items_by_lifecycle=popular_items_by_lifecycle,
        popular_items_by_behavior=popular_items_by_behavior,
    )

    logger.info("HybridRecommender ready")
    return recommender
