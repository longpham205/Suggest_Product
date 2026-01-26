
import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    PURCHASE_HISTORY_CSV_PATH,
    PRODUCTS_PATH,
    DEPARTMENTS_PATH,
    LIFECYCLE_ASSIGNMENTS_PATH,
    BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH,
    POPULAR_ITEMS_GLOBAL_PATH,
    POPULAR_ITEMS_BY_LIFECYCLE_PATH,
    POPULAR_ITEMS_BY_BEHAVIOR_PATH,
    PRODUCT_DEPARTMENT_PATH
)
from src.utils.io import save_pickle, save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_checkpoints():
    logger.info("Loading data...")
    
    # Load transactions
    if not PURCHASE_HISTORY_CSV_PATH.exists():
        logger.error(f"Transaction data not found at {PURCHASE_HISTORY_CSV_PATH}")
        return

    df_txns = pd.read_csv(PURCHASE_HISTORY_CSV_PATH, usecols=["user_id", "product_id"])
    
    # Load Metadata
    df_products = pd.read_csv(PRODUCTS_PATH, usecols=["product_id", "department_id"])
    df_depts = pd.read_csv(DEPARTMENTS_PATH)
    
    # Load User Assignments
    df_lifecycle = pd.read_csv(LIFECYCLE_ASSIGNMENTS_PATH)
    df_behavior = pd.read_csv(BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH)

    # 1. Product Department Map
    logger.info("Building Product-Department Map form JSON...")
    # Map department_id to department_name
    dept_map = df_depts.set_index("department_id")["department"].to_dict()
    
    # Map product_id to department_name
    prod_dept_map = {}
    for _, row in df_products.iterrows():
        pid = int(row["product_id"])
        did = int(row["department_id"])
        if did in dept_map:
            prod_dept_map[str(pid)] = dept_map[did]
            
    save_json(prod_dept_map, PRODUCT_DEPARTMENT_PATH)
    logger.info(f"Saved product_department_map to {PRODUCT_DEPARTMENT_PATH}")

    # 2. Global Popular Items
    logger.info("Calculating Global Popular items...")
    global_top_50 = df_txns["product_id"].value_counts().head(50).index.tolist()
    global_top_50 = [int(x) for x in global_top_50]
    
    save_pickle(global_top_50, POPULAR_ITEMS_GLOBAL_PATH)
    logger.info(f"Saved global popular items to {POPULAR_ITEMS_GLOBAL_PATH}")

    # 3. Lifecycle Popular Items
    logger.info("Calculating Popular Items by Lifecycle...")
    # Merge transactions with lifecycle
    df_merged_life = df_txns.merge(df_lifecycle, on="user_id", how="inner")
    
    lifecycle_popular = {}
    for stage in df_merged_life["lifecycle_stage"].unique():
        top_items = (
            df_merged_life[df_merged_life["lifecycle_stage"] == stage]["product_id"]
            .value_counts()
            .head(50)
            .index.tolist()
        )
        lifecycle_popular[str(stage)] = [int(x) for x in top_items]
        
    save_pickle(lifecycle_popular, POPULAR_ITEMS_BY_LIFECYCLE_PATH)
    logger.info(f"Saved items by lifecycle to {POPULAR_ITEMS_BY_LIFECYCLE_PATH}")

    # 4. Behavior Popular Items
    logger.info("Calculating Popular Items by Behavior Cluster...")
    # Merge transactions with behavior
    # Note: behavior csv usually has 'cluster' column, let's verify or rename
    if "behavior_cluster" not in df_behavior.columns:
        # Assuming 'cluster' is the column name for behavior cluster if not explicit
        df_behavior.rename(columns={"cluster": "behavior_cluster"}, inplace=True)

    df_merged_beh = df_txns.merge(df_behavior[["user_id", "behavior_cluster"]], on="user_id", how="inner")
    
    behavior_popular = {}
    for cluster in df_merged_beh["behavior_cluster"].unique():
        top_items = (
            df_merged_beh[df_merged_beh["behavior_cluster"] == cluster]["product_id"]
            .value_counts()
            .head(50)
            .index.tolist()
        )
        behavior_popular[int(cluster)] = [int(x) for x in top_items]
        
    save_pickle(behavior_popular, POPULAR_ITEMS_BY_BEHAVIOR_PATH)
    logger.info(f"Saved items by behavior to {POPULAR_ITEMS_BY_BEHAVIOR_PATH}")
    
    logger.info("DONE! All checkpoints rebuilt.")

if __name__ == "__main__":
    build_checkpoints()
