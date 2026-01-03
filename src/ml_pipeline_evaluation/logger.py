from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RunLogger:
    run_dir: str

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)

    def write_json(self, name: str, data: Dict[str, Any]) -> str:
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def write_metrics_csv(self, name: str, rows: Dict[str, float]) -> str:
        path = os.path.join(self.run_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in rows.items():
                writer.writerow([k, f"{v:.6f}" if isinstance(v, (int, float)) else str(v)])
        return path
