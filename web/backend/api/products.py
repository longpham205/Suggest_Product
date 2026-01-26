# web/backend/pi/products.py

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from web.backend.dependencies.model_loader import get_product_service

router = APIRouter(prefix="/api/products", tags=["products"])


# ==========================================================
# Response Schemas
# ==========================================================
class ProductItem(BaseModel):
    product_id: int
    product_name: str
    aisle_id: int
    department_id: int
    price: float = 100000


class ProductListResponse(BaseModel):
    products: List[ProductItem]
    total: int
    page: int
    page_size: int


# ==========================================================
# Endpoints
# ==========================================================
@router.get("", response_model=ProductListResponse)
def get_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    department_id: Optional[int] = None,
    search: Optional[str] = None,
    service=Depends(get_product_service),
):
    """
    Lấy danh sách sản phẩm với pagination và filter
    """
    products, total = service.get_products(
        page=page,
        page_size=page_size,
        department_id=department_id,
        search=search,
    )
    return {
        "products": products,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/departments/list")
def get_departments(
    service=Depends(get_product_service),
):
    """
    Lấy danh sách departments
    """
    return service.get_departments()


@router.get("/{product_id}")
def get_product(
    product_id: int,
    service=Depends(get_product_service),
):
    """
    Lấy thông tin chi tiết sản phẩm
    """
    product = service.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
