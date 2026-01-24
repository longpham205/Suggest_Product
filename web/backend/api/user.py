# api/user.py

from fastapi import APIRouter, Depends
from schemas.user import UserProfileResponse
from services.user_profile_service import UserProfileService
from dependencies.model_loader import get_user_profile_service

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/{user_id}", response_model=UserProfileResponse)
def get_user_profile(
    user_id: int,
    time_bucket: str,
    is_weekend: bool,
    service: UserProfileService = Depends(get_user_profile_service),
):
    """
    Get user profile & cluster info
    """

    # Build minimal context (same logic as recommender)
    user_context = {
        "behavior_cluster": 0,
        "preference_cluster": 0,
        "lifecycle_stage": "new",
    }

    return service.get_user_profile(
        user_id=user_id,
        user_context=user_context,
    )
