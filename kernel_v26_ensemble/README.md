# TF V26 — V22/V24 State Ensemble

The state-task `speed_kmh` and `flow_vph` predictions are the arithmetic mean of V22 and V24.
Queue labels and ODME path flows are retained from V24. The 6,980,503-row template order and
submission schema are preserved.

- Inputs: `kernel_v22/cloud_output/submission.csv`, `kernel_v24/cloud_output/submission.csv`
- Rebuild: `python traffic_flow/build_ensembles_v25_v27.py`
- Public score: pending
