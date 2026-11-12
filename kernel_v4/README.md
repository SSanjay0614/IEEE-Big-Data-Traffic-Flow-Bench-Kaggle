# V4 — top-rank heuristics on top of the same public assets

CPU only. Candidate #2 for the public-LB catch-up, forked 2026-10-27 from
[sujanmajhisuzan/trafficflowbench-top-rank-pipeline-0-90-target](https://www.kaggle.com/code/sujanmajhisuzan/trafficflowbench-top-rank-pipeline-0-90-target).

The author claims a 0.924+ target built on the same public assets as v3. Its incremental ideas
over the 0.86517 pipeline, worth keeping even if the kernel's own score disappoints:

1. **Structural onset invariant** — the organizer's window selector puts queue onsets only at
   T+30 (h=6); predicting anything at h<6 is noise, so only h=6 is modelled, with a top-k
   fallback because every onset window contains a queue by construction.
2. **Adaptive per-horizon thresholds** for ongoing windows, calibrated against horizon decay.
3. **1D spatial morphological closing** along neighbouring links to remove isolated predicted
   cells and reconnect split queues.
4. **Physical guardrails on Task 1** — speed clipped to ≥ 5 km/h so Task 3's derived density
   never divides by ~zero; flow clipped at ≥ 0.
5. **Dual warm-start ODME** — link-space warm start, then bounded L-BFGS-B with tight tolerances
   (maxiter 8000, ftol 1e-14), λ = 0.05.

## Status

Forked verbatim and pushed as `sanjay0614/tf-v4-top-rank` (GPU off). Score appended when graded.
