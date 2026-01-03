from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import joblib
import yaml
import numpy as np

from .data import DataConfig, load_dataset
from .pipeline import PipelineConfig, build_pipeline
from .metrics import MetricsConfig, compute_metrics
from .logger import RunLogger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a clean scikit-learn experiment with logging.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg["seed"])
    np.random.seed(seed)

    # Data
    dcfg = DataConfig(
        dataset=str(cfg["data"]["dataset"]),
        test_size=float(cfg["data"]["test_size"]),
        val_size=float(cfg["data"]["val_size"]),
        seed=seed,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dcfg)

    # Pipeline
    pcfg = PipelineConfig(
        scaler=str(cfg["pipeline"]["scaler"]),
        model=str(cfg["pipeline"]["model"]),
        logreg=dict(cfg["pipeline"].get("logreg", {})) or None,
        rf=dict(cfg["pipeline"].get("rf", {})) or None,
    )
    pipe = build_pipeline(pcfg)

    # Fit
    pipe.fit(X_train, y_train)

    # Predict
    def pred_all(X, y):
        y_pred = pipe.predict(X)
        y_prob = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_prob = pipe.predict_proba(X)[:, 1]
            except Exception:
                y_prob = None
        mcfg = MetricsConfig(primary=str(cfg["metrics"]["primary"]), report=list(cfg["metrics"]["report"]))
        metrics = compute_metrics(y, y_pred, y_prob, mcfg)
        return metrics

    m_train = pred_all(X_train, y_train)
    m_val = pred_all(X_val, y_val)
    m_test = pred_all(X_test, y_test)

    # Logging
    runs_dir = cfg["output"]["runs_dir"]
    artifacts_dir = cfg["output"]["artifacts_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{pcfg.model}-{pcfg.scaler}-{ts}"

    run_dir = os.path.join(runs_dir, run_name)
    logger = RunLogger(run_dir=run_dir)

    payload = {
        "run_name": run_name,
        "timestamp": ts,
        "seed": seed,
        "data": cfg["data"],
        "pipeline": cfg["pipeline"],
        "metrics": {
            "train": m_train,
            "val": m_val,
            "test": m_test,
        },
    }

    logger.write_json("metrics.json", payload)

    # Also emit a flat CSV for quick scanning
    flat = {}
    for split, ms in payload["metrics"].items():
        for k, v in ms.items():
            flat[f"{split}_{k}"] = v
    logger.write_metrics_csv("metrics.csv", flat)

    # Save model artifact
    model_path = os.path.join(artifacts_dir, run_name, "model.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    print(f"Run: {run_name}")
    print(f"Saved metrics to: {os.path.join(run_dir, 'metrics.json')} and metrics.csv")
    print(f"Saved model to: {model_path}")
    print("Test metrics:")
    for k, v in m_test.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
