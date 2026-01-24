# api/cart.py

from fastapi import APIRouter, Depends
from schemas.cart import CartAddRequest, CartAddResponse
from services.cart_service import CartService
from dependencies.model_loader import get_cart_service

router = APIRouter(prefix="/api/cart", tags=["cart"])


@router.post("/add", response_model=CartAddResponse)
def add_to_cart(
    req: CartAddRequest,
    service: CartService = Depends(get_cart_service),
):
    """
    Cart boost recommendation (stateless)
    """

    boosted_product = service.recommend_after_add(
        user_id=req.user_id,
        added_product_id=req.product_id,
        time_bucket=req.time_bucket,
        is_weekend=req.is_weekend,
    )

    return {
        "boosted_product_id": boosted_product
    }
