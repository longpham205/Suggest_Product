# web/backend/services/user_service.py

import pandas as pd
from typing import Dict, List
from src.config.settings import PURCHASE_HISTORY_CSV_PATH


# ================================
# Mapping cluster -> text (BE)
# ================================
BEHAVIOR_CLUSTER_TEXT = {
    0: "Người mua không thường xuyên",
    1: "Người mua ổn định",
    2: "Người mua nhiều",
    3: "Khách hàng tiềm năng",
    4: "Khách hàng VIP",
}

PREFERENCE_CLUSTER_TEXT = {
    0: "Ưa chuộng hàng thiết yếu",
    1: "Ưa chuộng hàng khuyến mãi",
    2: "Ưa chuộng sản phẩm cao cấp",
    3: "Ưa chuộng hàng mới",
    4: "Ưa chuộng hàng đa dạng",
}

LIFECYCLE_STAGE_TEXT = {
    "new": "Khách hàng mới",
    "regular": "Khách thường xuyên",
    "active": "Khách hàng đang hoạt động",
    "loyal": "Khách hàng trung thành",
    "vip": "Khách VIP",
    "churn_risk": "Có nguy cơ rời bỏ",
}


class UserProfileService:
    """
    Build user profile info for FE
    """

    def __init__(self):
        self.df = pd.read_csv(PURCHASE_HISTORY_CSV_PATH)

    # ======================================
    # Public API
    # ======================================
    def get_user_profile(
        self,
        user_id: int,
        user_context: Dict,
        recent_k: int = 5,
    ) -> Dict:
        """
        Return:
        {
            cluster_info: {...},
            recent_purchases: [product_id]
        }
        """

        recent_items = self._get_recent_purchases(user_id, recent_k)

        cluster_info = {
            "behavior_cluster": user_context["behavior_cluster"],
            "behavior_text": BEHAVIOR_CLUSTER_TEXT.get(
                int(user_context["behavior_cluster"]), "Không xác định"
            ),
            "preference_cluster": user_context["preference_cluster"],
            "preference_text": PREFERENCE_CLUSTER_TEXT.get(
                int(user_context["preference_cluster"]), "Không xác định"
            ),
            "lifecycle_stage": user_context["lifecycle_stage"],
            "lifecycle_text": LIFECYCLE_STAGE_TEXT.get(
                user_context["lifecycle_stage"], "Không xác định"
            ),
        }

        return {
            "user_id": user_id,
            "cluster_info": cluster_info,
            "recent_purchases": recent_items,
        }

    # ======================================
    # Internal helpers
    # ======================================
    def _get_recent_purchases(
        self,
        user_id: int,
        k: int,
    ) -> List[Dict]:

        user_df = self.df[self.df["user_id"] == user_id]

        if user_df.empty:
            return []

        # Sort by order_time if column exists
        if "order_time" in user_df.columns:
            user_df = user_df.sort_values("order_time", ascending=False)

        return (
            user_df[["product_id", "product_name"]]
            .assign(product_name=lambda x: x["product_name"].fillna(f"Sản phẩm #{x['product_id']}"))
            .head(k)
            .to_dict(orient="records")
        )

