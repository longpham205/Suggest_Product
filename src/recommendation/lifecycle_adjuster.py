# src/recommendation/lifecycle_adjuster.py

"""
Lifecycle-based score adjuster (STAGE-LEVEL)

- Adjust recommendation scores based on lifecycle_stage
- Lifecycle stage MUST be provided upstream (UserContextLoader)
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class LifecycleAdjuster:
    """
    Adjust recommendation scores based on lifecycle stage.

    Responsibilities:
    - Apply stage-specific score adjustment policy
    - Does NOT load data
    - Does NOT map user_id -> stage
    """

    STAGE_POLICIES = {
        "new": {
            "head_boost": 1.15,
            "tail_boost": 1.00,
        },
        "regular": {
            "head_boost": 1.00,
            "tail_boost": 1.00,
        },
        "loyal": {
            "head_boost": 0.95,
            "tail_boost": 1.10,
        },
    }

    # ======================================================
    # CORE LOGIC
    # ======================================================
    def adjust(
        self,
        scores: Dict[int, float],
        lifecycle_stage: str,
    ) -> Dict[int, float]:
        """
        Adjust candidate scores based on lifecycle stage.

        Args:
            scores: product_id -> base score
            lifecycle_stage: pre-assigned lifecycle stage

        Returns:
            Dict[product_id, adjusted score]
        """
        if not scores:
            return {}

        policy = self.STAGE_POLICIES.get(
            lifecycle_stage,
            self.STAGE_POLICIES["regular"],
        )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        n = len(ranked)
        head_cutoff = max(1, int(0.3 * n))  # top 30%

        adjusted: Dict[int, float] = {}

        for i, (pid, score) in enumerate(ranked):
            if i < head_cutoff:
                adjusted[pid] = score * policy["head_boost"]
            else:
                adjusted[pid] = score * policy["tail_boost"]

        logger.debug(
            f"LifecycleAdjuster applied | "
            f"stage={lifecycle_stage} | "
            f"head={head_cutoff} | "
            f"total={n}"
        )

        return adjusted
