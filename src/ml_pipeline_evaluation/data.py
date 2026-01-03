from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "breast_cancer"
    test_size: float = 0.2
    val_size: float = 0.2
    seed: int = 42


def load_dataset(cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset and return X_train, y_train, X_val, y_val, X_test, y_test."""
    if cfg.dataset != "breast_cancer":
        raise ValueError("Only 'breast_cancer' dataset is supported in this minimal repo.")

    data = load_breast_cancer()
    X = data.data
    y = data.target

    # First split out test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # Then split train/val from the remaining
    val_rel = cfg.val_size / max(1e-12, (1.0 - cfg.test_size))
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=cfg.seed, stratify=y_trainval
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
