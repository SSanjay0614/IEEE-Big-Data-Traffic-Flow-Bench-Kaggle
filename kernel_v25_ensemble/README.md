# TF V25 — V23/V24 State Ensemble

The state-task `speed_kmh` and `flow_vph` predictions are the arithmetic mean of V23 and V24.
Queue labels and ODME path flows are retained from V24. The 6,980,503-row template order and
submission schema are preserved.

- Inputs: `kernel_v23/cloud_output/submission.csv`, `kernel_v24/cloud_output/submission.csv`
- Rebuild: `python traffic_flow/build_ensembles_v25_v27.py`
- Public score: pending
