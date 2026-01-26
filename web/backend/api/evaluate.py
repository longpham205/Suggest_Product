# web/backend/api/evaluate.py
"""
Evaluation API endpoint for offline metrics
"""

import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional

from web.backend.dependencies.model_loader import get_recommend_service

router = APIRouter(tags=["evaluate"])
logger = logging.getLogger(__name__)

# Thread pool for running sync evaluation
eval_executor = ThreadPoolExecutor(max_workers=2)

# Timeout for evaluation (seconds)
EVAL_TIMEOUT = 300  # Increased to 5 minutes


class EvaluationRequest(BaseModel):
    """Request schema for evaluation"""
    max_users: int = 20  # Reduced default
    top_k: int = 10


class EvaluationResponse(BaseModel):
    """Response schema for evaluation metrics"""
    metrics: Dict[str, Any]
    status: str
    message: Optional[str] = None
    duration_seconds: Optional[float] = None


@router.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(
    req: EvaluationRequest,
    service=Depends(get_recommend_service),
):
    """
    Run offline evaluation on the recommendation system
    
    Returns metrics including:
    - Precision@K, Recall@K, HitRate@K
    - RuleUserCoverage, RuleItemShare
    - Level distribution (L1-L5)
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting evaluation: max_users={req.max_users}, top_k={req.top_k}")
        
        # Import here to avoid circular imports
        from src.evaluation.offline_eval import OfflineEvaluator
        
        loop = asyncio.get_event_loop()
        
        def run_eval():
            evaluator = OfflineEvaluator(
                recommender=service.recommender,
                max_users=req.max_users
            )
            return evaluator.evaluate(k=req.top_k)
        
        metrics = await asyncio.wait_for(
            loop.run_in_executor(eval_executor, run_eval),
            timeout=EVAL_TIMEOUT
        )
        
        duration = time.time() - start_time
        logger.info(f"Evaluation completed in {duration:.1f}s: {len(metrics)} metrics")
        
        return EvaluationResponse(
            metrics=metrics,
            status="success",
            message=f"Evaluated {metrics.get('num_users_evaluated', 0)} users",
            duration_seconds=round(duration, 1)
        )
        
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        logger.error(f"Evaluation timeout after {duration:.1f}s")
        return EvaluationResponse(
            metrics={},
            status="error",
            message=f"Timeout after {duration:.0f}s. Try reducing max_users to 10-20.",
            duration_seconds=round(duration, 1)
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Evaluation error: {e}")
        return EvaluationResponse(
            metrics={},
            status="error",
            message=str(e),
            duration_seconds=round(duration, 1)
        )


@router.get("/evaluate/quick", response_model=EvaluationResponse)
async def quick_evaluation(
    max_users: int = Query(default=10, le=100, description="Number of users to evaluate"),
    top_k: int = Query(default=10, le=50, description="Top-K recommendations"),
    service=Depends(get_recommend_service),
):
    """
    Quick evaluation with small defaults (GET endpoint for easy testing)
    Default: 10 users, top-10 recommendations
    """
    return await run_evaluation(
        EvaluationRequest(max_users=max_users, top_k=top_k),
        service=service
    )

