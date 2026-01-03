# Suggested ablations (for lab-style evaluation)

Try these by editing `configs/default.yaml`:

1) Model swap:
- `pipeline.model: logreg` -> `rf`

2) Scaling:
- `pipeline.scaler: standard` -> `minmax` -> `none`

3) Regularization:
- `pipeline.logreg.C`: 0.1, 1.0, 10.0

4) Random Forest capacity:
- `pipeline.rf.max_depth`: null, 6, 12

Each run produces comparable metrics JSON/CSV for easy comparison.
