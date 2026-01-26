# web/backend/schemas/recommend.py

from pydantic import BaseModel, Field
from typing import List, Optional


# =========================
# Request
# =========================
class RecommendRequest(BaseModel):
    """
    Input schema for recommendation API
    """

    user_id: int = Field(..., example=12345)

    purchase_history_path: str = Field(
        ...,
        description="Path to purchase history CSV file",
        example="data/purchase_history/user_12345.csv",
    )

    top_k: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of items to recommend (default=20)",
    )

    time_bucket: str = Field(..., example="evening")
    is_weekend: bool = Field(..., example=False)


# =========================
# Response item
# =========================
class RecommendItem(BaseModel):
    """
    Single recommended item
    """

    item_id: int = Field(..., example=10123)
    score: float = Field(..., example=0.87)
    source: List[str] = Field(
        ..., example=["RULE", "POPULAR"]
    )


# =========================
# Metadata
# =========================
class RecommendMetadata(BaseModel):
    """
    Debug / evaluation metadata
    """

    rule_candidates: int
    fallback_used: bool
    insurance_used: bool
    final_returned: int

    rule_item_count: Optional[int] = None
    popular_item_count: Optional[int] = None
    insurance_item_count: Optional[int] = None


# =========================
# Response
# =========================
class RecommendResponse(BaseModel):
    """
    Output schema for recommendation API
    """

    user_id: int
    recommendations: List[RecommendItem]
    metadata: Optional[RecommendMetadata] = None
