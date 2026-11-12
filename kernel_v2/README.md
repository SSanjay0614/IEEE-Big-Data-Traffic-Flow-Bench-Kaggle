# V2 — first complete submission (build)

CPU only. The pipeline check from the old V5 milestone, pulled forward on 2026-10-27 at the
user's go-ahead: with five submissions a day, the cheapest lesson — a complete file accepted and
scored on every task — is bought immediately, not on Nov 6.

## What it does

One kernel run produces `submission.csv`:

1. **state**: historical-mean profiles (v1's method) predicted on the `validation` and `private`
   Task 1 templates, panel by panel;
2. **queue**: the host's persistence baseline mirrored exactly (last history step at T-5,
   released `v_cut` threshold else 0.6 × free speed, repeated across all six steps);
3. **odme**: per panel/split ridge-regularised NNLS on the released link counts — the task here is
   one aggregated departure time per panel, so the systems are tiny and real NNLS costs nothing;
4. **merge**: the host's `merge_submissions.py` logic, streamed over `submission_key.csv`.

Hard assertions before the file is declared ready: exactly 6,980,503 rows, zero nulls, zero
unmatched keys, queue values in {0, 1}, all predicted values non-negative.

## Status

Built and pushed. Run result and any leaderboard number go here once known.
