# V5 — iterate and final submission (planned)

Everything after the pipeline is proven. Inputs available by then: v2's complete baseline
submission, v3's physics-aware state, v4's ODME tuning. Expected order of value:

1. **Task 1 accuracy** (0.35 + moves Task 3's 0.15) — v3's reconstruction submitted whatever it
   wins on `train`.
2. **Task 2** (0.30, baseline 0.2518) — physics forward-run: run the v3 reconstruction over the
   horizon and threshold at the queue cut; onset windows get an upstream-gradient detector.
   Judged by the public leaderboard only and only ten windows per panel per split, so moves here
   are bundled with other improvements, one submission at a time.
3. **Task 4** — v4's tuned ODME.

## Final days

Deadline 2026-11-07 06:55 UTC — the real cut-off is the evening of Nov 6 locally. Last day: no new
modelling, final selection between the two best submissions, confirm the chosen file is complete,
submit, stop.

## Status

Planned.
