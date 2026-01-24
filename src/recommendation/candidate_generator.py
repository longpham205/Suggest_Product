# src/recommendation/candidate_generator.py
"""
FP-Growth Candidate Generator with Hierarchical Context (L1 → L5)

✔ MATCH theo context_key THỰC TẾ đã train (sparse)
✔ KHÔNG build context cứng từ user
✔ Ưu tiên level L1 → L5
✔ GLOBAL fallback đúng nghĩa
✔ Weighted level decay
"""

import os
import sys
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Tuple

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT_DIR)

from src.association_rules.rule_store import load_fpgrowth_rule_index
from src.config.settings import (
    FPGROWTH_RULE_INDEX_PATH,
    FPGROWTH_MAX_ANTECEDENT_LEN,
    CONTEXT_HIERARCHY,
    DEFAULT_TOP_K,
    FPGROWTH_LEVEL_DECAY,
)

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    FP-Growth based candidate generator with hierarchical context fallback.

    Priority:
        L1 (most specific) → L5 (GLOBAL)
    """

    # ==============================================================
    # INIT
    # ==============================================================
    def __init__(
        self,
        rule_index_path: Path | None = None,
        max_antecedent_len: int = FPGROWTH_MAX_ANTECEDENT_LEN,
    ):
        self.rule_index_path = Path(rule_index_path or FPGROWTH_RULE_INDEX_PATH)
        self.max_antecedent_len = max_antecedent_len

        # level -> { context_key -> rule_index }
        self.rules_by_level: Dict[str, Dict[str, Dict]] = defaultdict(dict)

        # preserve priority L1 → L5
        self.context_levels = list(CONTEXT_HIERARCHY.keys())

        self._last_matched_contexts: List[str] = []

        self._load_rules()

        logger.info(
            f"CandidateGenerator initialized | "
            f"levels={self.context_levels} | "
            f"max_ant_len={self.max_antecedent_len}"
        )

    # ==============================================================
    # LOAD RULE INDEX
    # ==============================================================
    def _load_rules(self):
        if not self.rule_index_path.exists():
            raise FileNotFoundError(
                f"FP-Growth rule index not found: {self.rule_index_path}"
            )

        raw = load_fpgrowth_rule_index(self.rule_index_path)

        # unwrap preview if exists
        if isinstance(raw, dict) and "preview" in raw:
            raw = raw["preview"]

        # schema v2: { _meta, data }
        if isinstance(raw, dict) and "data" in raw:
            data = raw["data"]
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError("Invalid FP-Growth rule index format")

        for context_key, rule_index in data.items():
            if not isinstance(rule_index, dict):
                continue

            level = self._infer_level(context_key)
            self.rules_by_level[level][context_key] = rule_index

        logger.info(
            f"Loaded FP-Growth rules | contexts="
            f"{sum(len(v) for v in self.rules_by_level.values())}"
        )

    # ==============================================================
    # INFER LEVEL FROM CONTEXT KEY
    # ==============================================================
    @staticmethod
    def _infer_level(context_key: str) -> str:
        if context_key == "GLOBAL":
            return "L5"

        dims_in_key = {
            part.split("=", 1)[0]
            for part in context_key.split("|")
            if "=" in part
        }

        best_level = "L5"
        best_len = 0

        for level, dims in CONTEXT_HIERARCHY.items():
            dims_set = set(dims)
            if dims_set.issubset(dims_in_key) and len(dims_set) > best_len:
                best_level = level
                best_len = len(dims_set)

        return best_level

    # ==============================================================
    # CONTEXT MATCH (CORE FIX)
    # ==============================================================
    @staticmethod
    def _context_match(user_context: Dict[str, str], context_key: str) -> bool:
        """
        Match user_context với context_key đã train
        user có nhiều dim hơn vẫn match được
        """
        if context_key == "GLOBAL":
            return True

        for part in context_key.split("|"):
            k, v = part.split("=", 1)
            if user_context.get(k) != v:
                return False

        return True

    # ==============================================================
    # USER CONTEXT
    # ==============================================================
    def build_user_context(self, **kwargs) -> Dict[str, str]:
        """
        Convert user attributes to string EXACTLY like training
        """
        return {k: str(v) for k, v in kwargs.items() if v is not None}

    # ==============================================================
    # GENERATE ANTECEDENTS
    # ==============================================================
    def _generate_antecedents(self, basket: List[int]) -> List[str]:
        basket = sorted(set(basket))[:20]  # safety cap
        ants: List[str] = []

        max_len = min(len(basket), self.max_antecedent_len)
        for l in range(1, max_len + 1):
            for combo in combinations(basket, l):
                ants.append("|".join(map(str, combo)))

        return ants

    # ==============================================================
    # MAIN GENERATION (FIXED)
    # ==============================================================
    def generate(
        self,
        basket: List[int],
        user_context: Dict[str, str],
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[List[int], Dict[int, float], Dict[int, set]]:

        if not basket:
            return [], {}, {}

        basket = [int(x) for x in basket]
        antecedents = self._generate_antecedents(basket)

        final_scores = defaultdict(float)
        rule_sources = defaultdict(set)   # pid -> {"L1", "L2", ...}
        matched_contexts: List[str] = []

        # ==================================================
        # L1 → L5 hierarchical recall
        # ==================================================
        for level in self.context_levels:
            decay = FPGROWTH_LEVEL_DECAY.get(level, 1.0)
            level_hits = 0

            for ctx_key, rule_index in self.rules_by_level.get(level, {}).items():

                # --------------------------------------------------
                # CONTEXT FILTER
                # L1–L4: phải match context
                # L5: GLOBAL → bỏ qua context
                # --------------------------------------------------
                if level != "L5":
                    if not self._context_match(user_context, ctx_key):
                        continue

                ctx_hits = 0

                for ant in antecedents:
                    rules = rule_index.get(ant, [])
                    for r in rules:
                        pid = int(r["consequent"])
                        if pid in basket:
                            continue

                        score = float(r.get("score", 0.0))
                        final_scores[pid] += score * decay
                        rule_sources[pid].add(level)

                        ctx_hits += 1
                        level_hits += 1

                if ctx_hits > 0:
                    matched_contexts.append(
                        f"{level}::{ctx_key} (hits={ctx_hits}, decay={decay:.2f})"
                    )

            # nếu level này không recall được gì → tiếp tục fallback
            if level_hits == 0:
                continue

        self._last_matched_contexts = matched_contexts

        if not final_scores:
            return [], {}, {}

        # ==================================================
        # CUT TOP-K (ranking will re-rank later)
        # ==================================================
        ranked = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        candidates = [pid for pid, _ in ranked]
        rule_scores = dict(ranked)
        rule_sources = {pid: rule_sources[pid] for pid in candidates}

        return candidates, rule_scores, rule_sources


    # ==============================================================
    # DEBUG
    # ==============================================================
    def get_last_matched_contexts(self) -> List[str]:
        return self._last_matched_contexts
