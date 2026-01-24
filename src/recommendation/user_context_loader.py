# src/recommendation/user_context_loader.py

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config.settings import (
    BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH,
    PREFERENCE_CLUSTER_ASSIGNMENTS_PATH,
    LIFECYCLE_ASSIGNMENTS_PATH,
)

logger = logging.getLogger(__name__)


class UserContextLoader:
    """
    Load and provide user-level context information:

    - behavior_cluster
    - preference_cluster
    - lifecycle_stage

    Data is loaded once and cached in memory.
    """

    def __init__(
        self,
        default_behavior_cluster: int = -1,
        default_preference_cluster: int = -1,
        default_lifecycle_stage: str = "unknown",
    ):
        self.default_behavior_cluster = default_behavior_cluster
        self.default_preference_cluster = default_preference_cluster
        self.default_lifecycle_stage = default_lifecycle_stage

        self.behavior_map: Dict[int, int] = {}
        self.preference_map: Dict[int, int] = {}
        self.lifecycle_map: Dict[int, str] = {}

        self._load_all()

    # ======================================================
    # Public API
    # ======================================================
    def get_user_context(self, user_id: int) -> Dict[str, Any]:
        """
        Return user context by user_id

        Returns
        -------
        dict with keys:
        - behavior_cluster
        - preference_cluster
        - lifecycle_stage
        """

        return {
            "behavior_cluster": self.behavior_map.get(
                user_id, self.default_behavior_cluster
            ),
            "preference_cluster": self.preference_map.get(
                user_id, self.default_preference_cluster
            ),
            "lifecycle_stage": self.lifecycle_map.get(
                user_id, self.default_lifecycle_stage
            ),
        }

    # ======================================================
    # Internal loading
    # ======================================================
    def _load_all(self):
        self.behavior_map = self._load_cluster_map(
            path=BEHAVIOR_CLUSTER_ASSIGNMENTS_PATH,
            key_col="user_id",
            value_col="cluster",
            name="behavior",
        )

        self.preference_map = self._load_cluster_map(
            path=PREFERENCE_CLUSTER_ASSIGNMENTS_PATH,
            key_col="user_id",
            value_col="cluster",
            name="preference",
        )

        self.lifecycle_map = self._load_cluster_map(
            path=LIFECYCLE_ASSIGNMENTS_PATH,
            key_col="user_id",
            value_col="lifecycle_stage",
            name="lifecycle",
        )

    def _load_cluster_map(
        self,
        path: Path,
        key_col: str,
        value_col: str,
        name: str,
    ) -> Dict[Any, Any]:
        """
        Load CSV file into dict: user_id -> cluster/stage
        """

        if not path.exists():
            logger.warning(
                f"[UserContextLoader] {name} file not found: {path}"
            )
            return {}

        df = pd.read_csv(path)

        if key_col not in df.columns or value_col not in df.columns:
            raise ValueError(
                f"{name} file missing required columns: "
                f"{key_col}, {value_col}"
            )

        mapping = dict(zip(df[key_col], df[value_col]))

        logger.info(
            f"[UserContextLoader] Loaded {len(mapping):,} "
            f"{name} assignments from {path.name}"
        )

        return mapping
