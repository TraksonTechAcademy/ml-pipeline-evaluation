from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricsConfig:
    primary: str = "f1"
    report: List[str] = None

    def __post_init__(self):
        if self.report is None:
            object.__setattr__(self, "report", ["accuracy", "precision", "recall", "f1", "roc_auc"])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, cfg: MetricsConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "accuracy" in cfg.report:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "precision" in cfg.report:
        out["precision"] = float(precision_score(y_true, y_pred))
    if "recall" in cfg.report:
        out["recall"] = float(recall_score(y_true, y_pred))
    if "f1" in cfg.report:
        out["f1"] = float(f1_score(y_true, y_pred))
    if "roc_auc" in cfg.report and y_prob is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["roc_auc"] = float("nan")
    return out
