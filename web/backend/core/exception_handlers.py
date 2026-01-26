# web/backend/core/exception_handlers.py

"""
Global exception handlers for FastAPI.
Provides consistent JSON error responses for all custom exceptions.
"""

import logging
from fastapi import Request
from fastapi.responses import JSONResponse

from web.backend.core.exceptions import (
    RecommendationError,
    UserNotFoundError,
    ProductNotFoundError,
    EmptyBasketError,
    InvalidTimeBucketError,
    ModelNotLoadedError,
    DataFileNotFoundError,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Error Response Helper
# ==========================================================
def error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: dict = None
) -> JSONResponse:
    """Create consistent error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "type": error_type,
                "message": message,
                "details": details or {}
            }
        }
    )


# ==========================================================
# Exception Handlers
# ==========================================================
async def user_not_found_handler(request: Request, exc: UserNotFoundError):
    logger.warning(f"User not found: {exc.user_id}")
    return error_response(
        status_code=404,
        error_type="USER_NOT_FOUND",
        message=str(exc),
        details={"user_id": exc.user_id}
    )


async def product_not_found_handler(request: Request, exc: ProductNotFoundError):
    logger.warning(f"Product not found: {exc.product_id}")
    return error_response(
        status_code=404,
        error_type="PRODUCT_NOT_FOUND",
        message=str(exc),
        details={"product_id": exc.product_id}
    )


async def empty_basket_handler(request: Request, exc: EmptyBasketError):
    return error_response(
        status_code=400,
        error_type="EMPTY_BASKET",
        message="User has no purchase history. Cannot generate recommendations.",
    )


async def invalid_time_bucket_handler(request: Request, exc: InvalidTimeBucketError):
    return error_response(
        status_code=422,
        error_type="VALIDATION_ERROR",
        message=str(exc),
        details={"field": "time_bucket", "value": exc.value}
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    logger.error(f"Model not loaded: {exc}")
    return error_response(
        status_code=503,
        error_type="SERVICE_UNAVAILABLE",
        message="Recommendation model is not available. Please try again later.",
    )


async def data_file_not_found_handler(request: Request, exc: DataFileNotFoundError):
    logger.error(f"Data file missing: {exc.path}")
    return error_response(
        status_code=503,
        error_type="SERVICE_UNAVAILABLE",
        message="Required data files are missing. Contact administrator.",
    )


async def generic_recommendation_error_handler(request: Request, exc: RecommendationError):
    logger.error(f"Recommendation error: {exc}")
    return error_response(
        status_code=500,
        error_type="RECOMMENDATION_ERROR",
        message="An error occurred while generating recommendations.",
    )


async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return error_response(
        status_code=500,
        error_type="INTERNAL_ERROR",
        message="An unexpected error occurred. Please try again.",
    )


# ==========================================================
# Register all handlers
# ==========================================================
def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app"""
    app.add_exception_handler(UserNotFoundError, user_not_found_handler)
    app.add_exception_handler(ProductNotFoundError, product_not_found_handler)
    app.add_exception_handler(EmptyBasketError, empty_basket_handler)
    app.add_exception_handler(InvalidTimeBucketError, invalid_time_bucket_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    app.add_exception_handler(DataFileNotFoundError, data_file_not_found_handler)
    app.add_exception_handler(RecommendationError, generic_recommendation_error_handler)
    app.add_exception_handler(Exception, global_exception_handler)
