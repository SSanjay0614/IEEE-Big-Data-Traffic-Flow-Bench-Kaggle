# V3 — public pipeline adoption (0.86517 route)

CPU only. New direction, 2026-10-27: public notebooks score ~0.87 while ours scored 0.47761 —
adopting the community's best beats rebuilding in five days.

## The source notebook

[anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517](https://www.kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517),
ranked 48/320 when published. Forked verbatim into this folder as `tf_v3_public_pipeline.ipynb`,
pushed as `sanjay0614/tf-v3-public-pipeline`.

- **Task 1**: interpolation baseline + CatBoost depth-8/depth-10 residuals on 94 features,
  physical clipping on top.
- **Task 2**: causal ongoing LightGBM (36 features) + two-stage onset LightGBM (83 features) +
  GPU/CPU-portable CatBoost onset classifier, top-1 fallback.
- **Task 4**: nonnegative ridge with dual warm start, high-precision L-BFGS-B on each split's
  own counts and prior (λ = 0.05) — the author's largest single gain over the earlier public
  solves.
- Submission assembled on `submission_key.csv` order with hard assertions.

## Why this one and not the 0.87/trees-unet notebook

That one depends on the author's private model dataset (`reference_candW.parquet` ships in no
public dataset — probe confirmed; its run errored at cell 2). The 0.86517 solution publishes
EVERYTHING: all seven checkpoints in dataset
`anubisyy/trafficflowbench-086517-public-assets`, checksum-verified at startup, and CatBoost
predict auto-falls back to CPU — so this runs with the GPU disabled (the quota stays reserved
for ARCC-AGI-3 on Oct 31).

Possible top-ups from the 0.90-target kernel later: structural onset invariant, adaptive
horizon thresholds, spatial morphological closing.

## Status

Version 2 (the verbatim fork) running. Score appended to the action plan when graded.
