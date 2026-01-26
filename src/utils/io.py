# src/utils/io.py

"""
Utility functions for loading/saving data files.
"""

import json
import pickle
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load data from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_pickle(path: Path) -> Any:
    """Load data from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Save data to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(data, f)
