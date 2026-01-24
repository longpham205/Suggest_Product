# services/recommend_service.py

import pandas as pd
from typing import Dict, List

from src.config.settings import DEFAULT_TOP_K, PURCHASE_HISTORY_CSV_PATH
from src.recommendation.hybrid_recommender import HybridRecommender
from services.user_profile_service import UserProfileService


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
        self.df = pd.read_csv(PURCHASE_HISTORY_CSV_PATH)

    # ======================================
    # Public API
    # ======================================
    def recommend(
        self,
        user_id: int,
        time_bucket: str,
        is_weekend: bool,
    ) -> Dict:
        """
        Return payload for FE
        """

        basket = self._build_user_basket(user_id)

        # ---- Call model
        results, metadata = self.recommender.recommend(
            user_id=user_id,
            basket=basket,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
            top_k=DEFAULT_TOP_K,
            return_metadata=True,
        )

        # ---- User profile
        user_context = self.recommender._build_user_context(
            user_id=user_id,
            time_bucket=time_bucket,
            is_weekend=is_weekend,
        )

        user_profile = self.user_profile_service.get_user_profile(
            user_id=user_id,
            user_context=user_context,
        )

        return {
            "user": user_profile,
            "recommendations": results,
            "metadata": metadata,  # optional (debug / demo)
        }

    # ======================================
    # Internal
    # ======================================
    def _build_user_basket(self, user_id: int, k: int = 10) -> List[str]:
        """
        Basket = last k purchased product_ids (string)
        """

        user_df = self.df[self.df["user_id"] == user_id]

        if user_df.empty:
            return []

        user_df = user_df.sort_values("order_time", ascending=False)

        return user_df["product_id"].astype(str).head(k).tolist()
