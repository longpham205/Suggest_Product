# schemas/user.py

from pydantic import BaseModel, Field
from typing import Optional


class UserContext(BaseModel):
    """
    User context after clustering / enrichment
    (được load từ UserContextLoader)
    """

    user_id: int = Field(..., description="User ID")
    behavior_cluster: int = Field(..., description="Behavior cluster ID")
    preference_cluster: int = Field(..., description="Preference cluster ID")
    lifecycle_stage: str = Field(..., description="Lifecycle stage name")

    time_bucket: Optional[str] = Field(
        None, description="Time bucket (morning / afternoon / evening)"
    )
    is_weekend: Optional[bool] = Field(
        None, description="Is weekend flag"
    )


class UserRequest(BaseModel):
    """
    Raw user input from API
    """

    user_id: int = Field(..., example=12345)
    time_bucket: str = Field(..., example="evening")
    is_weekend: bool = Field(..., example=False)
