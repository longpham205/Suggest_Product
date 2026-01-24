# src/utils/spark_utils.py
"""
Spark Session Manager for FP-Growth (Context-Aware Association Rule Mining)

Design goals:
- Stable on Windows (local mode)
- Optimized for FP-Growth (memory-heavy, shuffle-heavy)
- Config-driven (via src.config.settings)
- Safe for repeated pipeline execution
- No business logic, no data logic
"""

from __future__ import annotations

import os
import sys
import platform
import logging
from pathlib import Path
from typing import Optional, Dict

from pyspark.sql import SparkSession

from src.config.settings import (
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_EXECUTOR_CORES,
    SPARK_SHUFFLE_PARTITIONS,
    SPARK_LOG_LEVEL,
    CHECKPOINT_DIR,
)

logger = logging.getLogger(__name__)


# ======================================================
# ENV DETECTION
# ======================================================

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _has_hadoop_home() -> bool:
    return bool(
        os.environ.get("HADOOP_HOME")
        or os.environ.get("hadoop.home.dir")
    )


# ======================================================
# Spark Session Factory
# ======================================================

def get_spark_session(
    app_name: str = SPARK_APP_NAME,
    driver_memory: str = SPARK_DRIVER_MEMORY,
    executor_memory: str = SPARK_EXECUTOR_MEMORY,
    executor_cores: int = SPARK_EXECUTOR_CORES,
    shuffle_partitions: int = SPARK_SHUFFLE_PARTITIONS,
    log_level: str = SPARK_LOG_LEVEL,
    *,
    enable_checkpoint: bool = False,
) -> SparkSession:
    """
    Create or get a SparkSession optimized for FP-Growth mining.
    Safe for Windows local mode.
    """

    # --------------------------------------------------
    # Windows-safe JAVA_HOME setup
    # --------------------------------------------------
    if _is_windows() and "JAVA_HOME" not in os.environ:
        java_candidates = [
            r"C:\Program Files\Java\jdk-17",
            r"C:\Program Files\Java\jdk-17.0.1",
            r"C:\Program Files\Java\jdk-17.0.2",
        ]
        for path in java_candidates:
            if os.path.exists(path):
                os.environ["JAVA_HOME"] = path
                logger.info(f"JAVA_HOME set to {path}")
                break

    # --------------------------------------------------
    # Ensure PySpark uses current Python environment
    # --------------------------------------------------
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # --------------------------------------------------
    # Parallelism tuning (FP-Growth friendly)
    # --------------------------------------------------
    cpu_count = os.cpu_count() or 8
    safe_shuffle_partitions = min(shuffle_partitions, cpu_count * 4)
    default_parallelism = min(cpu_count * 2, safe_shuffle_partitions)

    # --------------------------------------------------
    # Build SparkSession
    # --------------------------------------------------
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # ---- Memory
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.cores", str(executor_cores))
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.4")
        # ---- Shuffle & parallelism
        .config("spark.sql.shuffle.partitions", str(safe_shuffle_partitions))
        .config("spark.default.parallelism", str(default_parallelism))
        # ---- Serialization (CRITICAL for FP-Growth)
        .config(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer",
        )
        # ---- Adaptive execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # ---- Arrow
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            "true",
        )
        # ---- UI noise
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    # --------------------------------------------------
    # Spark runtime setup
    # --------------------------------------------------
    spark.sparkContext.setLogLevel(log_level)

    # --------------------------------------------------
    # CHECKPOINT (SAFE)
    # --------------------------------------------------
    if enable_checkpoint:
        if _is_windows() and not _has_hadoop_home():
            logger.warning(
                "Checkpoint DISABLED: Windows detected but "
                "HADOOP_HOME not configured."
            )
        else:
            checkpoint_dir = Path(CHECKPOINT_DIR) / "spark"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            spark.sparkContext.setCheckpointDir(
                str(checkpoint_dir.resolve())
            )

            logger.info(
                f"Spark checkpoint enabled at {checkpoint_dir}"
            )

    logger.info(
        "SparkSession initialized | "
        f"app={app_name} | "
        f"driver_memory={driver_memory} | "
        f"executor_memory={executor_memory} | "
        f"executor_cores={executor_cores} | "
        f"parallelism={default_parallelism} | "
        f"shuffle_partitions={safe_shuffle_partitions} | "
        f"spark_version={spark.version}"
    )

    return spark


# ======================================================
# Spark Utilities
# ======================================================

def stop_spark_session(spark: Optional[SparkSession]) -> None:
    """Safely stop SparkSession."""
    if spark is None:
        return

    try:
        spark.stop()
        logger.info("SparkSession stopped successfully")
    except Exception as exc:
        logger.warning(f"Error stopping SparkSession: {exc}")


def get_spark_context_info(spark: SparkSession) -> Dict[str, str]:
    """Retrieve useful SparkContext information."""
    sc = spark.sparkContext
    return {
        "spark_version": spark.version,
        "master": sc.master,
        "app_name": sc.appName,
        "default_parallelism": sc.defaultParallelism,
        "driver_memory": spark.conf.get("spark.driver.memory", "N/A"),
        "executor_memory": spark.conf.get("spark.executor.memory", "N/A"),
        "executor_cores": spark.conf.get("spark.executor.cores", "N/A"),
        "shuffle_partitions": spark.conf.get(
            "spark.sql.shuffle.partitions", "N/A"
        ),
        "checkpoint_dir": sc.getCheckpointDir() or "N/A",
    }


# ======================================================
# Smoke Test
# ======================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("Running SparkSession smoke test...")
    spark = get_spark_session(enable_checkpoint=False)

    info = get_spark_context_info(spark)
    for k, v in info.items():
        logger.info(f"{k}: {v}")

    df = spark.range(1_000_000)
    logger.info(f"Test DataFrame count: {df.count():,}")

    stop_spark_session(spark)
    logger.info("SparkSession smoke test completed successfully.")
