# src/association_rules/rule_store.py

"""
RuleStore – Persistence layer for context-aware association rules

Canonical format (schema v2, FP-Growth algorithm v2):
{
    "_meta": {
        "schema_version": 2,
        "algorithm": "fpgrowth",
        "algorithm_version": 2,
        "created_at": "...",
        "stats": {...}
    },
    "data": {
        context_key: {
            antecedent_key: [
                {
                    "rule_id": str,
                    "antecedent": list[str],
                    "consequent": int,
                    "confidence": float,
                    "lift": float,
                    "support": float,
                    "score": float
                }
            ]
        }
    }
}
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# =================================================
# TYPE ALIASES
# =================================================

Rule = Dict[str, Any]
RuleIndex = Dict[str, List[Rule]]
ContextRuleIndex = Dict[str, RuleIndex]

SCHEMA_VERSION = 2
SUPPORTED_ALGORITHMS = {"fpgrowth"}


# =================================================
# SAVE
# =================================================

def save_context_rule_index(
    context_rule_index: ContextRuleIndex,
    path: Path,
    *,
    algorithm: str = "fpgrowth",
    algorithm_version: int = 2,
) -> None:
    """
    Save context-aware rule index to disk (pickle, offline use).
    """

    if not context_rule_index:
        logger.warning("Context rule index is empty – nothing to save")
        return

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stats = _collect_stats(context_rule_index)

    payload = {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "algorithm": algorithm,
            "algorithm_version": algorithm_version,
            "created_at": datetime.utcnow().isoformat(),
            "stats": stats,
        },
        "data": context_rule_index,
    }

    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        logger.exception(f"Failed to save rule index to {path}")
        raise

    logger.info(
        "Rule index saved | "
        f"algo={algorithm} v{algorithm_version} | "
        f"contexts={stats['contexts']:,} | "
        f"antecedents={stats['antecedents']:,} | "
        f"rules={stats['rules']:,} | "
        f"path={path}"
    )


# =================================================
# LOAD
# =================================================

def load_context_rule_index(
    path: Path,
) -> Tuple[ContextRuleIndex, Dict[str, Any]]:
    """
    Load context-aware rule index.

    Returns
    -------
    (context_rule_index, meta)
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rule index file not found: {path}")

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        logger.exception(f"Failed to load rule index from {path}")
        raise

    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError("Invalid rule index payload (missing 'data')")

    meta = payload.get("_meta", {})
    context_rule_index = payload["data"]

    _validate_meta(meta)
    _validate_context_rule_index(context_rule_index)

    stats = meta.get("stats") or _collect_stats(context_rule_index)

    logger.info(
        "Rule index loaded | "
        f"algo={meta.get('algorithm')} v{meta.get('algorithm_version')} | "
        f"schema_v={meta.get('schema_version')} | "
        f"contexts={stats['contexts']:,} | "
        f"antecedents={stats['antecedents']:,} | "
        f"rules={stats['rules']:,} | "
        f"path={path}"
    )

    return context_rule_index, meta


# =================================================
# VALIDATION
# =================================================

def _validate_meta(meta: Dict[str, Any]) -> None:
    schema_v = meta.get("schema_version")
    if schema_v != SCHEMA_VERSION:
        logger.warning(
            f"Schema version mismatch: expected={SCHEMA_VERSION}, got={schema_v}"
        )

    algo = meta.get("algorithm")
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm in index: {algo}")


def _validate_context_rule_index(
    context_rule_index: ContextRuleIndex,
) -> None:
    """
    Light schema validation (sample-based).
    """

    if not isinstance(context_rule_index, dict):
        raise ValueError("context_rule_index must be dict")

    for ctx, rule_index in list(context_rule_index.items())[:3]:
        if not isinstance(ctx, str):
            raise ValueError("Context key must be str")

        if not isinstance(rule_index, dict):
            raise ValueError("RuleIndex must be dict")

        for ant_key, rules in list(rule_index.items())[:3]:
            if not isinstance(ant_key, str):
                raise ValueError("Antecedent key must be str")

            if not isinstance(rules, list):
                raise ValueError("Rules must be list")

            for rule in rules[:3]:
                _validate_rule(rule)


def _validate_rule(rule: Rule) -> None:
    required = {
        "rule_id",
        "antecedent",
        "consequent",
        "confidence",
        "lift",
        "support",
        "score",
    }

    if not isinstance(rule, dict):
        raise ValueError("Rule must be dict")

    missing = required - set(rule.keys())
    if missing:
        raise ValueError(f"Rule missing keys: {missing}")

    if not isinstance(rule["antecedent"], list):
        raise ValueError("Rule antecedent must be list")


# =================================================
# STATS
# =================================================

def _collect_stats(context_rule_index: ContextRuleIndex) -> Dict[str, int]:
    contexts = len(context_rule_index)
    antecedents = sum(len(v) for v in context_rule_index.values())
    rules = sum(
        len(rules)
        for rule_index in context_rule_index.values()
        for rules in rule_index.values()
    )

    return {
        "contexts": contexts,
        "antecedents": antecedents,
        "rules": rules,
    }


# =================================================
# CONVENIENCE
# =================================================

def load_fpgrowth_rule_index(
    path: Path,
) -> ContextRuleIndex:
    """
    Convenience loader for FP-Growth rule index.
    """
    context_rule_index, meta = load_context_rule_index(path)

    algo = meta.get("algorithm")
    if algo != "fpgrowth":
        logger.warning(
            f"Expected FP-Growth index, got algorithm={algo}"
        )

    return context_rule_index
