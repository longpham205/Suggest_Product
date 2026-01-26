# web/backend/api/cart.py

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from web.backend.services.cart_service import CartService
from web.backend.dependencies.model_loader import get_cart_service, get_product_service

router = APIRouter(tags=["cart"])
logger = logging.getLogger(__name__)


# ==========================================================
# FE-Compatible Request/Response Schemas
# ==========================================================
class FEContext(BaseModel):
    """Context from frontend (nested structure)"""
    time_bucket: str
    is_weekend: bool


class FECartBoostRequest(BaseModel):
    """Request schema matching frontend payload"""
    user_id: int
    added_product_id: int  # FE uses this field name
    context: FEContext
    cart_items: Optional[List[int]] = []


class FECartBoostResponse(BaseModel):
    """Response schema for cart boost"""
    boosted_product: Optional[Dict[str, Any]] = None


# ==========================================================
# API Endpoints
# ==========================================================
@router.post("/cart/boost", response_model=FECartBoostResponse)
def cart_boost(
    req: FECartBoostRequest,
    cart_service: CartService = Depends(get_cart_service),
    product_service=Depends(get_product_service),
):
    """
    Cart boost recommendation API (FE-compatible path)
    
    When user adds product to cart, suggest a related product.
    """
    try:
        logger.info(f"Cart boost for user {req.user_id}, added product {req.added_product_id}")
        
        boosted_id = cart_service.recommend_after_add(
            user_id=req.user_id,
            added_product_id=req.added_product_id,
            time_bucket=req.context.time_bucket,
            is_weekend=req.context.is_weekend,
        )

        if boosted_id:
            # Get product info from ProductService
            product = product_service.get_product_by_id(boosted_id)
            if product:
                return {"boosted_product": product}

        return {"boosted_product": None}

    except Exception as e:
        logger.error(f"Cart boost error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
