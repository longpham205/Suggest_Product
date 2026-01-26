# web/backend/services/product_service.py

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

from src.config.settings import PRODUCTS_PATH, DEPARTMENTS_PATH

logger = logging.getLogger(__name__)


class ProductService:
    """Service for product data operations"""

    def __init__(self):
        self.df = self._load_products()
        self.departments = self._load_departments()
        logger.info(f"Loaded {len(self.df)} products")

    def _load_products(self) -> pd.DataFrame:
        """Load products from CSV"""
        df = pd.read_csv(PRODUCTS_PATH)
        # Add default price (can be replaced with real price data)
        df["price"] = 100000  # Default 100k VND
        return df

    def _load_departments(self) -> Dict[int, str]:
        """Load departments from CSV"""
        try:
            dept_df = pd.read_csv(DEPARTMENTS_PATH)
            return dict(zip(dept_df["department_id"], dept_df["department"]))
        except Exception as e:
            logger.warning(f"Could not load departments: {e}")
            return {}

    def get_products(
        self,
        page: int = 1,
        page_size: int = 20,
        department_id: Optional[int] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[Dict], int]:
        """Get paginated products with optional filters"""

        df = self.df.copy()

        # Filter by department
        if department_id:
            df = df[df["department_id"] == department_id]

        # Search by name
        if search:
            df = df[df["product_name"].str.contains(search, case=False, na=False)]

        total = len(df)

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        df = df.iloc[start:end]

        products = df.to_dict("records")
        return products, total

    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """Get single product by ID"""
        row = self.df[self.df["product_id"] == product_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_departments(self) -> List[Dict]:
        """Get all departments"""
        return [
            {"department_id": k, "department_name": v}
            for k, v in self.departments.items()
        ]
