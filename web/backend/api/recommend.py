# api/recommend.py

from fastapi import APIRouter, Depends
from schemas.recommend import RecommendRequest, RecommendResponse
from services.recommend_service import RecommendService
from dependencies.model_loader import get_recommend_service

router = APIRouter(prefix="/api", tags=["recommend"])


@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    req: RecommendRequest,
    service: RecommendService = Depends(get_recommend_service),
):
    """
    Main recommendation API
    """

    result = service.recommend(
        user_id=req.user_id,
        time_bucket=req.time_bucket,
        is_weekend=req.is_weekend,
    )

    return result

