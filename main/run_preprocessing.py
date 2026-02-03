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
    
    logger.info("Building transactions context features...")
    build_transactions_context(
        save_parquet=True,
        sample_ratio=None,
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
