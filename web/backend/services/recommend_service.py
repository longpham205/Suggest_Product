#web/backend/services/recommend_service.py
import pandas as pd
import logging
from typing import Dict, List
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from src.config.settings import DEFAULT_TOP_K, PURCHASE_HISTORY_CSV_PATH
from src.recommendation.hybrid_recommender import HybridRecommender
from web.backend.services.user_service import UserProfileService

logger = logging.getLogger(__name__)

# Cache: 1000 items, 5 minutes TTL
recommend_cache = TTLCache(maxsize=1000, ttl=300)


class RecommendService:
    """
    Main recommendation orchestration
    """

    def __init__(
        self,
        recommender: HybridRecommender,
        user_profile_service: UserProfileService,
    ):
        self.recommender = recommender
        self.user_profile_service = user_profile_service
        self._product_service = None  # Lazy load
        
        try:
            self.df = pd.read_csv(PURCHASE_HISTORY_CSV_PATH)
            logger.info(f"Loaded {len(self.df)} purchase records")
        except Exception as e:
            logger.warning(f"Could not load purchase history: {e}")
            self.df = pd.DataFrame()

    @property
    def product_service(self):
        """Lazy load ProductService to avoid circular imports"""
        if self._product_service is None:
            from services.product_service import ProductService
            self._product_service = ProductService()
        return self._product_service

    # ======================================
    # Public API
    # ======================================
    @cached(cache=recommend_cache, key=lambda self, user_id, time_bucket, is_weekend: hashkey(user_id, time_bucket, is_weekend))
    def recommend(
        self,
        user_id: int,
        time_bucket: str,
        is_weekend: bool,
    ) -> Dict:
        """
        Return payload for FE with enriched product data
        """
        
        logger.info(f"=== RecommendService.recommend START (Cache Miss) ===")
        logger.info(f"user_id={user_id}, time_bucket={time_bucket}, is_weekend={is_weekend}")

        basket = self._build_user_basket(user_id)
        logger.info(f"User {user_id} basket: {basket[:5] if basket else 'EMPTY'}... (total: {len(basket)})")

        # Check if basket is empty - this is a common cause for L5 fallback
        if not basket:
            logger.warning(f"User {user_id} has EMPTY basket -> efficient rule matching unlikely")

        # ---- Call model
        results, metadata = self.recommender.recommend(
            user_id=user_id,
            basket=basket,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
            top_k=DEFAULT_TOP_K,
            return_metadata=True,
        )
        
        logger.info(f"HybridRecommender returned {len(results)} items")
        logger.info(f"Metadata: {metadata}")
        
        # Log source distribution for debugging
        sources = [r.get('source', []) for r in results]
        levels = [r.get('context_level', []) for r in results]
        logger.info(f"Sources distribution: {sources[:3]}...")
        logger.info(f"Levels distribution: {levels[:3]}...")

        # ---- Enrich with product names
        enriched_results = self._enrich_products(results)

        # ---- User profile
        user_context = self.recommender._build_user_context(
            user_id=user_id,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
        )
        
        logger.info(f"User context: {user_context}")

        user_profile = self.user_profile_service.get_user_profile(
            user_id=user_id,
            user_context=user_context,
        )

        logger.info(f"=== RecommendService.recommend END ===")
        
        return {
            "user": user_profile,
            "recommendations": enriched_results,
            "metadata": metadata,
        }

    # ======================================
    # Internal
    # ======================================
    def _build_user_basket(self, user_id: int, k: int = 10) -> List[str]:
        """
        Basket = last k purchased product_ids (string)
        """
        if self.df.empty:
            return []

        user_df = self.df[self.df["user_id"] == user_id]

        if user_df.empty:
            return []

        if "order_time" in user_df.columns:
            user_df = user_df.sort_values("order_time", ascending=False)

        return user_df["product_id"].astype(str).head(k).tolist()

    def _enrich_products(self, results: List[Dict]) -> List[Dict]:
        """
        Add product_name and price to each recommendation item
        """
        enriched = []
        
        for item in results:
            item_id = item.get("item_id")
            if item_id:
                product = self.product_service.get_product_by_id(int(item_id))
                if product:
                    item["product_name"] = product.get("product_name", f"Sản phẩm #{item_id}")
                    item["price"] = product.get("price", 100000)
                    item["department_id"] = product.get("department_id")
                else:
                    item["product_name"] = f"Sản phẩm #{item_id}"
                    item["price"] = 100000
            enriched.append(item)
        
        return enriched

