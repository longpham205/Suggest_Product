# main/run_preprocessing.py

import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing.load_data import load_raw_data
from src.preprocessing.clean_data import clean_instacart_data
from src.preprocessing.merge_data import merge_instacart_data

from src.preprocessing.build_behavior_features import build_behavior_features
from src.preprocessing.build_preference_features import build_preference_features
from src.preprocessing.build_lifecycle_features import build_lifecycle_features
from src.preprocessing.build_transactions_context import build_transactions_context
from src.scripts.build_popularity_checkpoints import build_checkpoints 


from src.config import settings

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    logger.info("=" * 60)
    logger.info("START PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # =====================================================
    # LOAD RAW DATA (SOURCE OF TRUTH)
    # =====================================================
    raw = load_raw_data()
    for name, df in raw.items():
        logger.info(f"Loaded raw {name}: {df.shape}")

    # =====================================================
    # CLEAN DATA (ONLY FOR EDA / DEBUG)
    # =====================================================
    logger.info("Cleaning raw data ")
    cleaned = clean_instacart_data(raw)
    for name, df in cleaned.items():
        logger.info(f"Cleaned {name}: {df.shape}")

    # =====================================================
    # MERGE DATA (OPTIONAL – FOR ANALYSIS / REPORT)
    # =====================================================
    logger.info("Merging datasets for analysis...")
    merged = merge_instacart_data(
        orders=cleaned["orders"],
        order_products=cleaned["order_products"],
        products=cleaned["products"],
        aisles=cleaned["aisles"],
        departments=cleaned["departments"]
    )
    merged.to_csv(settings.MERGED_DATA_PATH, index=False)
    logger.info(f"Merged data saved: {merged.shape}")

    # =====================================================
    # BUILD FEATURES (ALWAYS USE RAW DATA)
    # =====================================================

    logger.info("Building behavior features...")
    build_behavior_features(
        orders=raw["orders"],
        order_products=raw["order_products"]
    ).to_csv(settings.BEHAVIOR_FEATURES_PATH, index=False)

    logger.info("Building preference features...")
    build_preference_features(
        orders=raw["orders"],
        order_products=raw["order_products"],
        products=raw["products"],
        departments=raw["departments"]
    ).to_csv(settings.PREFERENCE_FEATURES_PATH, index=False)

    logger.info("Building lifecycle features...")
    build_lifecycle_features(
        orders=raw["orders"]
    ).to_csv(settings.LIFECYCLE_FEATURES_PATH, index=False)

    logger.info("Building context features...")
    build_transactions_context(
        raw["orders"],
        raw["order_products"]
    ).to_csv(settings.TRANSACTIONS_CONTEXT_PATH,   index=False )
    
    # =====================================================
    # EXPORT DATA FOR WEB BACKEND
    # =====================================================
    if getattr(settings, "ENABLE_WEB_EXPORT", False):
        logger.info("Exporting data for web backend...")
        build_checkpoints()

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE FINISHED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
