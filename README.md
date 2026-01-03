# ml-pipeline-evaluation — Clean ML Pipeline + Logging + Metrics (Reproducible)

This repo demonstrates **research-grade ML engineering** (scikit-learn):

- Clean, modular **pipeline** (preprocess → model)
- **Reproducible experiments** (seed + YAML config)
- Proper **train/val/test split**
- **Metrics logging** (JSON + CSV) and saved artifacts
- Baselines: **Logistic Regression** and **Random Forest**

> Designed to be high-signal for AI labs: correctness, reproducibility, evaluation discipline.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run an experiment:
```bash
python -m ml_pipeline_evaluation.run --config configs/default.yaml --run_name baseline-logreg
```

Outputs:
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/metrics.csv`
- `artifacts/<run_name>/model.joblib`
- `artifacts/<run_name>/preprocess.joblib` (if applicable)

---

## Why this is useful (Prof. Kim-style)

This repo shows you can:
- design **clean evaluation protocols**
- avoid leakage (split discipline)
- log metrics consistently
- produce reusable artifacts
- run ablations by modifying config

---

## Repo structure

```
ml-pipeline-evaluation/
  configs/default.yaml
  src/ml_pipeline_evaluation/
    data.py
    pipeline.py
    metrics.py
    logger.py
    run.py
  suites/
    ablations.md
```

---

## Extending to DL / Agentic AI
- Replace `pipeline.py` model with a neural model wrapper (PyTorch) and keep:
  - split discipline
  - metrics logging
  - artifact saving
- Add stress tests / shift scenarios to evaluate robustness.

---

## License
MIT
