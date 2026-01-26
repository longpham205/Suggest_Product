# web/backend/api/recommend.py

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from web.backend.services.recommend_service import RecommendService
from web.backend.dependencies.model_loader import get_recommend_service, get_product_service

router = APIRouter(tags=["recommend"])
logger = logging.getLogger(__name__)

# Thread pool for running sync code
executor = ThreadPoolExecutor(max_workers=4)

# Timeout for recommendation (seconds)
RECOMMEND_TIMEOUT = 10


# ==========================================================
# FE-Compatible Request/Response Schemas
# ==========================================================
class FEContext(BaseModel):
    """Context from frontend (nested structure)"""
    time_bucket: str
    is_weekend: bool


class FERecommendRequest(BaseModel):
    """Request schema matching frontend payload"""
    user_id: int
    context: FEContext
    cart_items: Optional[List[Any]] = []  # Accept both string and int


class FERecommendResponse(BaseModel):
    """Response schema matching frontend expectation"""
    recommended_products: List[Dict[str, Any]]
    user: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# ==========================================================
# API Endpoints
# ==========================================================
@router.post("/recommend", response_model=FERecommendResponse)
async def recommend(
    req: FERecommendRequest,
    service: RecommendService = Depends(get_recommend_service),
    product_service=Depends(get_product_service),
):
    """
    Recommendation API - Uses HybridRecommender with timeout fallback
    
    Flow:
    1. Try HybridRecommender (personalized recommendations)
    2. If timeout or error -> fallback to popular products
    """
    try:
        logger.info(f"Recommend request for user {req.user_id}, context: {req.context}")
        
        try:
            # Run recommendation in thread pool with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: service.recommend(
                        user_id=req.user_id,
                        time_bucket=req.context.time_bucket,
                        is_weekend=req.context.is_weekend,
                    )
                ),
                timeout=RECOMMEND_TIMEOUT
            )
            
            logger.info(f"Got {len(result.get('recommendations', []))} recommendations for user {req.user_id}")
            
            return {
                "recommended_products": result.get("recommendations", []),
                "user": result.get("user"),
                "metadata": result.get("metadata"),
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Recommendation timeout for user {req.user_id}, using fallback")
            return await _fallback_recommendations(req, product_service)
            
        except Exception as e:
            logger.error(f"Recommendation error for user {req.user_id}: {e}")
            return await _fallback_recommendations(req, product_service)

    except Exception as e:
        logger.error(f"Critical recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _fallback_recommendations(req: FERecommendRequest, product_service) -> Dict:
    """
    Fallback to popular products when HybridRecommender fails/times out
    """
    logger.info(f"Using fallback recommendations for user {req.user_id}")
    
    products, total = product_service.get_products(page=1, page_size=10)
    
    recommendations = []
    for i, product in enumerate(products):
        recommendations.append({
            "item_id": product["product_id"],
            "product_name": product["product_name"],
            "price": product.get("price", 100000),
            "department_id": product["department_id"],
            "score": round(0.95 - i * 0.05, 2),
            "source": ["POPULAR", "FALLBACK"]
        })
    
    user_info = {
        "user_id": req.user_id,
        "cluster_info": {
            "behavior_cluster": 0,
            "preference_cluster": 0,
            "lifecycle_stage": "unknown",
            "behavior_text": "Chưa xác định",
            "lifecycle_text": "Khách mới"
        },
        "recent_purchases": []
    }
    
    return {
        "recommended_products": recommendations,
        "user": user_info,
        "metadata": {
            "time_bucket": req.context.time_bucket,
            "is_weekend": req.context.is_weekend,
            "total_products": total,
            "source": "fallback"
        }
    }


