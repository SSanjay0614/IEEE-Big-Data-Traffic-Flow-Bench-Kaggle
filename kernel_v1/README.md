# V1 — Task 1 historical-mean baseline, reproduced and scored

CPU only. Progress check, not a submission candidate.

## What it does

1. Reads `corridors/<PANEL>/train/mainline_states/` and `mainline_states_masked/`.
2. Builds the competition's published baseline: one weekday-by-time-of-day speed and flow
   profile per link, learned from `train`, replayed at every masked cell.
3. Scores itself on `train` by joining the masked view back to the unmasked view, exactly as the
   organisers describe, and reports `S_speed`, `S_flow` and `S_state` per panel and regime.

## Why this first

Task 1 plus Task 3 is 0.50 of the total score, and Task 1 is the only task we can score locally.
Everything downstream — the physics term, and any honest comparison between two methods — rests
on having this harness working and trusted.

## What it deliberately does not do

- No queue model. Task 2 labels are withheld for every split, so it cannot be validated here.
- No ODME. Task 4 is a separate inverse problem and gets its own version.
- No submission file. Passing 6.98 million rows around before the numbers are trusted is wasted
  effort.

## Inputs

Competition data only: `2026-ieee-big-data-traffic-flow-bench`, about 12.4 GB unpacked. One
corridor at a time to stay inside the notebook disk and time budget.

## Output

- `v1_task1_scores.json` — per-panel, per-regime `S_speed`, `S_flow`, `S_state`.
- Printed confusion of our own scoring against the published baseline numbers.

## Result

Pending — not run yet.
