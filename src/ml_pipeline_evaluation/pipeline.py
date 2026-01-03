from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class PipelineConfig:
    scaler: str = "standard"  # standard | minmax | none
    model: str = "logreg"     # logreg | rf
    logreg: Optional[Dict[str, Any]] = None
    rf: Optional[Dict[str, Any]] = None


def build_pipeline(cfg: PipelineConfig) -> Pipeline:
    steps = []

    if cfg.scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif cfg.scaler == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    elif cfg.scaler == "none":
        pass
    else:
        raise ValueError(f"Unknown scaler: {cfg.scaler}")

    if cfg.model == "logreg":
        params = cfg.logreg or {"C": 1.0, "max_iter": 500}
        clf = LogisticRegression(**params)
    elif cfg.model == "rf":
        params = cfg.rf or {"n_estimators": 300, "max_depth": None}
        clf = RandomForestClassifier(**params, random_state=42)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    steps.append(("model", clf))
    return Pipeline(steps)
