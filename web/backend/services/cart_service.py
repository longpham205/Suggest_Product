# web/backend/services/cart_service.py

from typing import Optional
from src.recommendation.hybrid_recommender import HybridRecommender


class CartService:
    """
    Cart enhancement logic (stateless)
    """

    def __init__(self, recommender: HybridRecommender):
        self.recommender = recommender

    def recommend_after_add(
        self,
        user_id: int,
        added_product_id: int,
        time_bucket: str,
        is_weekend: bool,
    ) -> Optional[int]:
        """
        Return 1 related product_id (or None)
        """

        # Fake basket = just added item
        basket = [str(added_product_id)]

        results = self.recommender.recommend(
            user_id=user_id,
            basket=basket,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
            top_k=5,
        )

        for pid in results:
            if pid != added_product_id:
                return pid

        return None
