# schemas/cart.py

from pydantic import BaseModel, Field
from typing import List


class CartItem(BaseModel):
    """
    Optional cart item schema
    (Currently NOT used in backend)
    """

    product_id: int = Field(..., example=1001)
    quantity: int = Field(1, ge=1, example=2)


class Cart(BaseModel):
    """
    Cart schema – reserved for future use
    """

    items: List[CartItem] = Field(default_factory=list)
