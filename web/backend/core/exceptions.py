# exceptions.py

"""
Custom exceptions for the recommendation system.
All exceptions inherit from RecommendationError base class.
"""


# ==========================================================
# Base Exception
# ==========================================================
class RecommendationError(Exception):
    """Base exception for recommendation system"""
    pass


# ==========================================================
# User-related Exceptions
# ==========================================================
class UserNotFoundError(RecommendationError):
    """User ID không tồn tại trong hệ thống"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        super().__init__(f"User {user_id} not found")


class UserContextError(RecommendationError):
    """Lỗi khi load user context (cluster, lifecycle)"""
    pass


# ==========================================================
# Product-related Exceptions
# ==========================================================
class ProductNotFoundError(RecommendationError):
    """Product ID không tồn tại"""
    def __init__(self, product_id: int):
        self.product_id = product_id
        super().__init__(f"Product {product_id} not found")


class EmptyBasketError(RecommendationError):
    """Basket rỗng, không thể gợi ý"""
    pass


# ==========================================================
# Model/Data Exceptions
# ==========================================================
class ModelNotLoadedError(RecommendationError):
    """Model chưa được load hoặc bị lỗi"""
    pass


class DataFileNotFoundError(RecommendationError):
    """File dữ liệu (CSV, pickle) không tìm thấy"""
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Data file not found: {path}")


class RuleIndexError(RecommendationError):
    """Lỗi khi query association rules"""
    pass


# ==========================================================
# Validation Exceptions
# ==========================================================
class InvalidTimeBucketError(RecommendationError):
    """time_bucket không hợp lệ"""
    VALID_VALUES = ["morning", "afternoon", "evening", "night"]
    
    def __init__(self, value: str):
        self.value = value
        super().__init__(
            f"Invalid time_bucket '{value}'. Valid: {self.VALID_VALUES}"
        )


class InvalidTopKError(RecommendationError):
    """top_k nằm ngoài phạm vi cho phép"""
    pass
