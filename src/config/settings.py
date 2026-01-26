# src/config/settings.py
from pathlib import Path

# =================================================
# PROJECT ROOT
# =================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =================================================
# DATA DIRECTORIES
# =================================================
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# RAW DATA FILES
# =================================================
AISLES_PATH = DATA_RAW_DIR / "aisles.csv"
DEPARTMENTS_PATH = DATA_RAW_DIR / "departments.csv"
PRODUCTS_PATH = DATA_RAW_DIR / "products.csv"
ORDERS_PATH = DATA_RAW_DIR / "orders.csv"
ORDER_PRIOR_PATH = DATA_RAW_DIR / "order_products_prior.csv"
ORDER_TRAIN_PATH = DATA_RAW_DIR / "order_products_train.csv"

# =================================================
# PROCESSED DATA FILES
# =================================================
MERGED_DATA_PATH = DATA_PROCESSED_DIR / "merged_data.csv"

BEHAVIOR_FEATURES_PATH = DATA_PROCESSED_DIR / "behavior_features.csv"
PREFERENCE_FEATURES_PATH = DATA_PROCESSED_DIR / "preference_features.csv"
LIFECYCLE_FEATURES_PATH = DATA_PROCESSED_DIR / "lifecycle_features.csv"

TRANSACTIONS_CONTEXT_PATH = DATA_PROCESSED_DIR / "transactions_context.csv"

# Purchase history for recommendation service
PURCHASE_HISTORY_CSV_PATH = DATA_PROCESSED_DIR / "merged_data.csv"

# =================================================
# CHECKPOINTS
# =================================================
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Behavior clustering ----
BEHAVIOR_CHECKPOINT_DIR = CHECKPOINT_DIR / "behavior"
BEHAVIOR_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_CLUSTER_MODEL_PATH = BEHAVIOR_CHECKPOINT_DIR / "behavior_clustering.pkl"
BEHAVIOR_SCALER_PATH = BEHAVIOR_CHECKPOINT_DIR / "behavior_scaler.pkl"

BEHAVIOR_CLUSTER_SCORE_PATH = BEHAVIOR_CHECKPOINT_DIR / "behavior_cluster_department_scores.pkl"

# ---- Preference clustering ----
PREFERENCE_CHECKPOINT_DIR = CHECKPOINT_DIR / "preference"
PREFERENCE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PREFERENCE_CLUSTER_MODEL_PATH = (
    PREFERENCE_CHECKPOINT_DIR / "preference_clustering.pkl"
)
PREFERENCE_SCALER_PATH = (
    PREFERENCE_CHECKPOINT_DIR / "preference_scaler.pkl"
)

PREFERENCE_CLUSTER_SCORE_PATH = (
    PREFERENCE_CHECKPOINT_DIR / "preference_cluster_department_scores.pkl"
)

# ---- Product/Popular items for recommendation ----
RECOMMENDATION_CHECKPOINT_DIR = CHECKPOINT_DIR / "recommendation"
RECOMMENDATION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT_DEPARTMENT_PATH = RECOMMENDATION_CHECKPOINT_DIR / "product_department_map.json"
POPULAR_ITEMS_GLOBAL_PATH = RECOMMENDATION_CHECKPOINT_DIR / "popular_items_global.pkl"
POPULAR_ITEMS_BY_LIFECYCLE_PATH = RECOMMENDATION_CHECKPOINT_DIR / "popular_items_by_lifecycle.pkl"
POPULAR_ITEMS_BY_BEHAVIOR_PATH = RECOMMENDATION_CHECKPOINT_DIR / "popular_items_by_behavior.pkl"

# ---- Spark (Parquet) ----
TRANSACTIONS_CONTEXT_EXTENDED_PATH = (
    DATA_PROCESSED_DIR / "transactions_context_extended.parquet"
)

# =================================================
# RESULTS
# =================================================
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# BEHAVIOR

BEHAVIOR_RESULTS_DIR = RESULTS_DIR / "behavior"
BEHAVIOR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOR_CLUSTER_REPORT_PATH = (
    BEHAVIOR_RESULTS_DIR / "behavior_cluster_report.txt"
)
BEHAVIOR_CLUSTER_TABLE_PATH = (
    BEHAVIOR_RESULTS_DIR / "behavior_cluster_summary.csv"
)
BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH = RESULTS_DIR / "behavior" / "behavior_cluster_assignments.csv"

# PREFERENCE

PREFERENCE_RESULTS_DIR = RESULTS_DIR / "preference"
PREFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREFERENCE_CLUSTER_REPORT_PATH = (
    PREFERENCE_RESULTS_DIR / "preference_cluster_report.txt"
)
PREFERENCE_CLUSTER_TABLE_PATH = (
    PREFERENCE_RESULTS_DIR / "preference_cluster_summary.csv"
)
PREFERENCE_CLUSTER_ASSIGNMENTS_PATH = RESULTS_DIR / "preference" / "preference_cluster_assignments.csv"

# LIFECYCLE

LIFECYCLE_ASSIGNMENTS_PATH = RESULTS_DIR / "lifecycle" / "lifecycle_assignments.csv"

# TRANSACTIONS

TRANSACTIONS_RESULTS_DIR = RESULTS_DIR / "transactions"
TRANSACTIONS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTIONS_RESULTS_CSV_PATH = TRANSACTIONS_RESULTS_DIR / "miner_stats.csv"
TRANSACTIONS_RESULTS_JSONL_PATH = TRANSACTIONS_RESULTS_DIR / "miner_stats.jsonl"

# EVALUATE

OFFLINE_EVALUATION_DIR = RESULTS_DIR / "evaluate"
OFFLINE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# =================================================
# CLUSTERING CONFIG
# =================================================
BEHAVIOR_CLUSTER_FEATURES = [
    "total_orders",
    "total_products",
    "reorder_ratio",
    "avg_days_between_orders",
]

N_BEHAVIOR_CLUSTERS = 5
N_PREFERENCE_CLUSTERS = 5
RANDOM_STATE = 42

# =================================================
# ASSOCIATION RULES – COMMON
# =================================================
RULE_INDEX_DIR = CHECKPOINT_DIR / "association_rules"
RULE_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# =================================================
# FP-GROWTH (SPARK) CONFIG
# =================================================

# ---- Thresholds ----
FPGROWTH_MIN_SUPPORT = 0.002
FPGROWTH_MIN_CONFIDENCE = 0.10
FPGROWTH_MIN_LIFT = 1.05

# ---- Rule constraints ----
FPGROWTH_MAX_ANTECEDENT_LEN = 3
FPGROWTH_MAX_RULES_PER_ANTECEDENT = 10
FPGROWTH_MAX_ITEMS_PER_TXN = 20

# ---- Rule index ----
FPGROWTH_RULE_INDEX_PATH = (
    RULE_INDEX_DIR / "fpgrowth_context_rule_index.pkl"
)

# =================================================
# RULE SCORING
# =================================================
RULE_SCORE_METHOD = "confidence_lift_weighted"
RULE_SCORE_CONF_WEIGHT = 0.7
RULE_SCORE_LIFT_WEIGHT = 0.3
MIN_RULE_SCORE = 0.0

# =================================================
# CONTEXT CONFIGURATION
# =================================================
CONTEXT_USE_TIME_BUCKET = True
CONTEXT_USE_WEEKEND = True
CONTEXT_USE_DOW = False

CONTEXT_KEY_FIELDS = [
    "time_bucket",
    "is_weekend",
]

CONTEXT_MIN_TXNS = 5000
MAX_RULES_PER_CONTEXT = 3000
DEFAULT_TOP_K = 20

# =========================
# EXTENDED CONTEXT CONFIGURATION (8 DIMENSIONS)
# =========================

# All available context dimensions
CONTEXT_DIMENSIONS_ALL = [
    # Temporal dimensions (4)
    "time_bucket",           # night/morning/afternoon/evening
    "is_weekend",            # 0/1
    "day_of_week",           # 0-6 (Mon-Sun)
    "basket_size_category",  # small/medium/large
    
    # User segment dimensions (4)
    "purchase_frequency",    # low/medium/high
    "lifecycle_stage",       # new/regular/loyal/vip
    "preference_cluster",    # 0-4
    "behavior_cluster",      # 0-4
]

# Hierarchical context levels (specific to general)
CONTEXT_HIERARCHY = {
    # Level 1: Most specific (5 dimensions)
    "L1": [
        "time_bucket", "is_weekend", 
        #"day_of_week", "basket_size_category", "purchase_frequency",
        "lifecycle_stage", "preference_cluster", "behavior_cluster"
    ],
    
    # Level 2: Remove day_of_week and basket_size (4 dimensions)
    "L2": [
        "time_bucket", "is_weekend", 
        "lifecycle_stage", "preference_cluster"
    ],
    
    # Level 3: Temporal + lifecycle only (3 dimensions)
    "L3": [
        "time_bucket", "is_weekend", "lifecycle_stage"
    ],
    
    # Level 4: Temporal only (2 dimensions)
    "L4": [
        "time_bucket", "is_weekend"
    ],
    
    # Level 5: Global fallback (no context)
    "L5": []
}

# Minimum transactions per context level
CONTEXT_MIN_TXNS_BY_LEVEL = {
    "L1": 500,    # Specific contexts need fewer txns
    "L2": 1000,
    "L3": 2000,
    "L4": 5000,
    "L5": 0,      # Global always exists
}
# Weight decay when falling back from specific
FPGROWTH_LEVEL_DECAY = {
    "L1": 2.0,    # Tăng mạnh L1 (specific nhất)
    "L2": 1.8,    # Tăng mạnh L2
    "L3": 1.5,    # Tăng L3
    "L4": 1.2,    # Tăng L4
    "L5": 0.3,    # Giảm mạnh L5/GLOBAL
}

# Basket size thresholds
BASKET_SIZE_BINS = [0, 5, 15, float("inf")]
BASKET_SIZE_LABELS = ["small", "medium", "large"]

# Purchase frequency thresholds (will be calculated from quantiles)
PURCHASE_FREQ_QUANTILES = [0.33, 0.67]
PURCHASE_FREQ_LABELS = ["low", "medium", "high"]

# =================================================
# SPARK RUNTIME CONFIG
# =================================================

SPARK_APP_NAME = "SuggestProduct-FPGrowth"
SPARK_LOG_LEVEL = "WARN"

SPARK_DRIVER_MEMORY = "6g"
SPARK_EXECUTOR_MEMORY = "6g"
SPARK_EXECUTOR_CORES = 2

SPARK_SHUFFLE_PARTITIONS = 200
SPARK_DEFAULT_PARALLELISM = 200

# =================================================
# VERSIONING
# =================================================
CONTEXT_VERSION = "v2"
RULE_INDEX_VERSION = 2

# =================================================
# EVALUATION
# =================================================
MAX_EVAL_USERS = 1000
MIN_RULE_CANDIDATES = 3
