# src/association_rules/miner_stats.py
import csv
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

from src.config.settings import (
    TRANSACTIONS_RESULTS_JSONL_PATH,
    TRANSACTIONS_RESULTS_CSV_PATH
)

FIELDS = [
    "timestamp",
    "context",
    "num_transactions",
    "min_support",
    "min_confidence",
    "num_frequent_itemsets",
    "num_rules",
    "avg_confidence",
    "max_confidence",
    "runtime_seconds",
    "status",
    "error"
]


def save_miner_stats(
    *,
    context: str,
    num_transactions: int,
    min_support: float,
    min_confidence: float,
    num_frequent_itemsets: int = 0,
    num_rules: int = 0,
    avg_confidence: float = 0.0,
    max_confidence: float = 0.0,
    runtime_seconds: float = 0.0,
    status: str = "success",
    error: Optional[str] = None
) -> None:
    """
    Lưu thống kê FP-Growth ra CSV + JSONL để theo dõi & vẽ biểu đồ
    """

    # Ensure output directories exist
    os.makedirs(os.path.dirname(TRANSACTIONS_RESULTS_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRANSACTIONS_RESULTS_JSONL_PATH), exist_ok=True)

    record: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "context": context,
        "num_transactions": num_transactions,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "num_frequent_itemsets": num_frequent_itemsets,
        "num_rules": num_rules,
        "avg_confidence": round(avg_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "runtime_seconds": round(runtime_seconds, 2),
        "status": status,
        "error": error or ""
    }

    # --- CSV ---
    write_header = not os.path.exists(TRANSACTIONS_RESULTS_CSV_PATH)
    with open(TRANSACTIONS_RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    # --- JSONL ---
    with open(TRANSACTIONS_RESULTS_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
