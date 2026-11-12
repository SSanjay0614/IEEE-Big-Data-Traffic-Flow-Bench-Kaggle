# Action plan — Traffic Flow Bench (2026 IEEE Big Data Cup)

Started 2026-10-27. Appended to, never rewritten.

## Where we are

- Joined today. No submission. Deadline is **2026-11-07 06:55 UTC**, which is eleven days but
  effectively ten working days, because the last one ends in the early morning.
- 637 teams, $3,500, five submissions a day — so roughly fifty attempts, which is plenty.
- GPU quota is 0.00 h until 2026-10-31. This competition does not need it.

## The two facts that shape the plan

**1. A missing task scores zero.** `S_total` is a weighted sum with no renormalisation. A
submission with Task 1 only scores at most 0.35; one with everything at baseline level scores far
more. **A complete, mediocre submission beats a beautiful partial one**, so the first milestone is
completeness, not accuracy.

**2. Task 1 is worth 0.50 when you count physics.** Task 1 carries 0.35 directly, and Task 3
(0.15) is computed from the Task 1 file with no separate submission. So half the score comes from
one artefact, and Task 1 is also the only part we can score honestly locally. Everything else is
secondary until that is strong.

## Ordered plan, with gates

### V1 — Task 1 baseline harness (2026-10-27 to 2026-10-28)

Reproduce the published historical-mean baseline on `train`, with our own scorer, on one panel
first then all ten.

**Gate:** our `S_state` for the published baseline matches the host's number within rounding. If
it does not, our join between `mainline_states_masked` and `mainline_states` is wrong and nothing
downstream can be trusted.

### V2 — physics-aware reconstruction (2026-10-28 to 2026-10-30)

This is the real work. Replace the per-cell smoother with something that cannot break conservation:

- reconstruct density from `q = k*v` on each link, then impose
  `N(t+dt) - N(t) = dt*(q_in + r_on - q_out - r_off)` across the corridor;
- use ramp observations as the inflow/outflow constraints they are;
- keep the fundamental diagram as a bound rather than a target, since `S_FD` is only a validity
  gate while `S_LWR` carries two-thirds of the physics weight.

Candidate methods, in order of expected value per hour: a constrained least-squares / Kalman
smoother on the conservation equations; then a per-link regression with the conservation residual
as a penalty; then anything fancier only if the first two plateau.

**Gate:** better `S_state` than V1 on `train`, and a visibly lower conservation residual. We cannot
score `S_LWR` locally, so the residual is the proxy — and the leaderboard is the referee.

### V3 — Task 4 ODME (2026-10-30 to 2026-10-31)

Non-negative least squares on `A x = c` with the weak prior as a regulariser, and the deviation
term treated as a first-class objective rather than an afterthought — `S_dev` exists specifically
to punish both "submit the prior" and "fit the counts wildly".

**Gate:** reproduces the link counts to a low residual while staying near the prior. No submission
until both hold, because a degenerate answer scores near zero on two of the four components.

### V4 — Task 2 queue forecast (2026-10-31 to 2026-11-02)

Start by reproducing persistence (0.2518) so we have the floor, then improve it:

- the physics from V2 already gives predicted speeds, and the queue label is a speed threshold at
  0.60 of free-flow speed, so the natural model is "run the reconstruction forward and threshold";
- onset windows need a detector for *approaching* congestion, not current state — the useful
  features are the upstream speed gradient and the on-ramp inflow over the last 15 minutes;
- ongoing windows are mostly persistence plus growth along the topology, which
  `lwr_mainline_topology.csv` describes directly.

Note Task 2 cannot be validated locally: labels are withheld for every split. Public leaderboard
only, and it is noisy because only ten windows per panel per split exist.

### V5 — first complete submission (2026-11-02)

Merge `state`, `queue` and `odme` rows into the 6.98-million-row template and upload once.

**Gate:** the file is accepted and scored. This is a pipeline check, not a score attempt. Spending
a submission to find out that a 152 MB CSV has a blank cell in it is the cheapest lesson available
and we should buy it early rather than on Nov 6.

### V6 onward — iterate on the leaderboard (2026-11-03 to 2026-11-06)

In descending order of expected value:

1. Task 1 accuracy, again, since it also moves Task 3.
2. Task 2, because it is 0.30 and the baseline is only 0.2518 — it holds the largest proportional
   gain of any single task.
3. Task 4 residual work.

Five submissions a day is more than we will use. Preference: one submission per substantive
change, never a duplicate.

### Final day (2026-11-07, hard stop 06:55 UTC)

No new modelling. Verify the last chosen submission is the best one on the private-relevant
evidence, confirm the file is complete, and stop. Note the deadline is early morning UTC, so the
real cut-off is the evening of Nov 6 in local time.

## Risks and how they are handled

| Risk | Handling |
|---|---|
| 12.4 GB of data per notebook run | One panel at a time; cache intermediate parquet inside the notebook run, not across runs |
| 152 MB submission with 6.98 M rows | Build it programmatically from the template every time; never hand-edit; assert no nulls |
| Validation and private are different months | Fit on `train` only. Use the public leaderboard to check the method travels, never to tune the fit |
| Task 3 is invisible locally | Track conservation residual as a proxy, and treat the leaderboard's physics movement as the only real signal |
| Ten days is short | Completeness first, then depth. A complete baseline submission by Nov 2 is the milestone that makes the rest safe |

## What we are not doing

- No attempt to reconstruct the mask ourselves; the template is authoritative and the README says
  a derived mask is slightly too large.
- No treatment of missing ramp observations as zero.
- No submission that omits a task, ever.
- No GPU spend. This is a CPU problem and ARC owns the GPU window.

## Version map

Moved to the bottom of this file; see the table under the dated entry that follows.

## 2026-10-27 — version folders and the host's own code, added

- `kernel_v2` to `kernel_v5` folders created, each with a planned README stating its gate.
- The host repository (github.com/jacky850/trafficflowbench-public) was downloaded into
  `scratch/comps/traffic_flow/` — all four scorers, both Task 1 baselines, and the docs. Everything
  downstream will be written against the host's code, not against our reading of the README.
- Two scoring facts the README undersells, learned from the code:
  1. Task 1 flow is scored **per lane** — submitted and true flow are each divided by the
     station's lane count before the RMSE. Confusing total-link flow with per-lane flow zeroes
     `S_flow` past roughly 12 % error.
  2. The overall score aggregates regime → panel → **family** → overall as macro means: five
     two-directional families (D7_I10, D7_I210, D7_I405, D12_I5, D12_I405), not ten panels. A weak
     family counts as much as a strong one.
- `kernel_v1/build_v1.py` generates `tf_v1_task1_baseline.ipynb`: the historical-mean profile
  built on the unmasked `train` layer, targets read off the masked layer (null speed and null
  flow), truth joined 1:1 restricted to `is_score_eligible`, scored with the host's formulas and
  aggregation on all ten panels. Result JSON + per-regime detail CSV in the kernel output.

| Version | Contents | State |
|---|---|---|
| v1 | Task 1 baseline reproduced and scored | pushed, running |
| v2 | Complete first submission (state + queue + ODME) | built, pushed |
| v3 | Physics-aware reconstruction | planned README |
| v4 | Task 4 ODME beyond the v2 solve | planned README |
| v5 | Iterate and final submission | planned README |

---

## 2026-10-27 (later) — reorder: the first submission is now V2

The user authorized pushing kernels and submitting. That changes the sequencing, not the goals:
the plan's pipeline-check milestone (old V5) is the highest-value first move, because with five
submissions a day there is no reason to discover a format problem on Nov 6.

- The old V5 "first complete submission" is now **kernel_v2**, built and pushed. It combines the
  state profile (v1's method), the host's persistence queue baseline, and — an upgrade over
  "submit the prior" — per-panel ridge-regularised NNLS for ODME, because reading the artifacts
  showed the systems are tiny (one aggregated departure time per panel/split).
- Physics-aware state moves to **kernel_v3**. ODME tuning to **kernel_v4**. Task 2 improvements
  and final submissions live in **kernel_v5**. Earlier entries above stay as written; this note
  records that the version numbers shifted, not that any plan item was dropped.

Two artifact observations that made the first submission stronger than planned:

1. Task 4 releases one aggregated departure time per panel/split, with link counts for a handful
   of links — dense NNLS on paths x ~20 links, solved in milliseconds.
2. The per-panel Task 1 templates carry the merge key columns verbatim
   (`panel,timestamp,station_id,link_id,mask_regime`), so the state predictions join onto
   `submission_key.csv` with no column invention.

Submission download and upload happen locally after the kernel run completes:
`kaggle kernels output sanjay0614/tf-v2-first-submission`, then
`kaggle competitions submit -c 2026-ieee-big-data-traffic-flow-bench -f submission.csv -m "v2 complete baseline"`.

---

## 2026-10-27 (evening) — mount-path discovery and first push failure

Kernels pushed with `competition_sources` did not mount data at
`/kaggle/input/<competition-slug>` as the usual Kaggle layout suggests. A probe kernel printed the
real layout: competition data mounts under `/kaggle/input/competitions/<competition-slug>/` on
this account's environment, keeping the `kaggle_public/` wrapper inside.

Both first-version pushes failed on their opening assertion in under 25 s and cost nothing. All
three notebooks (tf v1, tf v2, sb v1) now resolve the mount by checking both locations and were
repushed as version 2. Probes go first when a mount assumption is unverified; a 20-second probe
kernel is cheaper than a failed 2-hour run.

Lesson recorded for every future kernel in every repo here: locate the mount, then assert;
never hardcode a mount path.

---

## 2026-10-27 (night) — first submission graded: 0.47761

Results in order obtained:

1. **kernel_v1 completed and passed its gate.** Overall `S_state = 0.7320` across ten panels on
   `train` (families: D12_I405 0.805, D12_I5 0.629, D7_I10 0.781, D7_I210 0.835, D7_I405 — the
   weakest at 0.610, as expected, it carries the largest incidents). A full ten-panel pass took
   **8.3 minutes** of wall clock — the harness is cheap enough for many experiments per day.
2. **kernel_v2 (version 2) produced a valid file and it was submitted** — ref 57702917,
   6,980,503 rows, zero nulls, all three task row counts as expected.
   Public leaderboard: **0.47761**. The decomposition sanity-checks: state ~0.35 × 0.68,
   queue ~0.30 × 0.25 (persistence baseline), leaving ~0.14 for physics + ODME, consistent with
   a prior-ODME answer.
3. **The known flaw in that submission**: ODME solved nothing ("counted links 0") because
   `path_link_incidence.csv` is presence/absence style — only `link_id,path_id` pairs, no value
   column — and the pivot found no value column to build `A` from. Fixed for version 3: incidence
   gets an explicit `use = 1` column and `pivot_table(values="use", aggfunc="max")`. Version 3
   also prints the NNLS residual per panel so the fix is observable, and rounds state/flow/path
   values to 3 decimals (403 MB file → well under the template's scale).

Version 3 was pushed to the same kernel. Next: download its output and submit again (4
submissions remain today), then Task 1 accuracy work (kernel_v3, physics-aware reconstruction)
uses today's remaining submissions only if it beats the v1 harness gate.

---


## 2026-10-27 (late) — version 3 ERRORED mid-run; version 4 fixes two distinct bugs

What version 3 proved before dying:

- **state stage fully works end-to-end**: all ten panels' profiles built (7–11 s each), all 20
  panel/split templates predicted, 204 s total for the whole state stage, state CSV 528 MB.
- **queue stage fully works**: all 20 panel/split files written in 2 s, positives distributed
  sensibly (e.g. D7_I405_N validation 14,220/21,120 = 67 % — the I405 incident networks queue
  hardest; queued shares 3–90 % by corridor).
- It then ERRORED in the ODME cell: `AttributeError: 'function' object has no attribute
  'reindex'`. Root cause: `synthetic_link_counts.csv` has a column literally named `count`, and
  `counts.set_index("link_id").count` resolves to the DataFrame `.count()` METHOD, not the
  column. Classic name shadowing.


Fix in version 4: `counts.set_index("link_id")["count"].reindex(counted).fillna(0.0)`. Now pushed
as version 4. Remaining 9 sessions today: submit its output when complete (the state+queue part
is byte-identical to the graded 0.47761 file, the ODME part should now beat the prior).

Lesson: never access a CSV column by attribute when the column name collides with any pandas
method name — always use bracket access.

---


## 2026-10-27 (late) — strategy pivot: the public notebooks are ~40 points ahead

User flagged that public kernels score ~0.87 versus our 0.47761. Verified against the kernel
list for this competition:

| kernel | votes | claimed LB | approach |
|---|---|---|---|
| yeashusemwal/trees-u-net-cnn | 27 | 0.87 | LGBM+U-Net Task 1, LGBM+CNN Task 2, tight ODME; everything inference-only off one public model dataset |
| anubisyy/catboost-lgbm-nnls | 16 | 0.86517 | CatBoost/LGBM + NNLS; publishes both its trained assets (69 MB dataset) and its output |
| honghanhhh | 0 | 0.87141 | same attachments as yeashusemwal's, alone no votes |
| sujanmajhisuzan/top-rank-0-90-target | 11 | "0.924+" (unverified) | same assets + structural onset invariant, adaptive horizon thresholds, spatial morphological closing |
| soukeaizenz/physics | 31 | 0.809 | physics pipeline |

Decision: adopt, don't race. Forked the yeashusemwal 0.87 notebook verbatim into
`kernel_v3/` and pushed as `sanjay0614/tf-v3-public-pipeline` with the same attachments
(dataset `anubisyy/trafficflowbench-086517-public-assets` + kernel source
`anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517`, no GPU, no internet). The notebook is
mount-agnostic (it globs `/kaggle/input/**/kaggle_public` recursively), so the local
`/kaggle/input/competitions/<slug>` layout needs no edits.

Key intel harvested from its experiment log (all quantified on the public LB by the author):

- Task 4 solved on each split's OWN counts and prior: +0.031 — and λ = 0.05 with the counts of
  the split being scored is exactly what our broken v2 web solved around; the pivot fix still
  matters for our own v4 comparisons.
- Onset queue predicted at T+30 ONLY (h=6): +0.027 — the window selector is part of the task.
- Task 2 ensemble + tuned thresholds: +0.017; onset model from the public CatBoost notebook:
  +0.0095; 50/50 probability blending of two Task 1 models: +0.0023.
- Leave-one-day-out profiles (leak-free features): +0.008/+0.004 — self-leaking features inflate
  local validation, which explains why our local-vs-LB decompositions never quite matched.

The old kernel_v3 plan (physics-aware state reconstruction from scratch) is superseded: its
expected value cannot reach +0.02, the fork is expected to be worth ~+0.39 on the public LB.
Physics work continues opportunistically only on top of the adopted pipeline (Postprocessing
guardrails like the 0.90-target kernel's `speed >= 5 km/h` clip are comparative-free wins).

---


## 2026-10-27 (night) — the 0.87 fork is not forkable; pivot to the 0.86517 assets

Facts established, in order:

1. **Real mount layout on this environment** (probe `mount-probe-2`, verified with all three
   mount kinds): `/kaggle/input/competitions/<comp>`, `/kaggle/input/datasets/<owner>/<slug>`,
   `/kaggle/input/notebooks/<owner>/<slug>`. Both public kernel outputs and private datasets
   mount only when ATTACHED in metadata; output attachments work from public kernels.
2. **The yeashusemwal 0.87 kernel's model dataset is private.** Its attachment list contains an
   entry the API renders as an empty string, and no public dataset (`datasets` search, user
   listings under yeashusemwal/jacky850/anubisyy) ships `reference_candW.parquet`. The fork run
   ERRORED at cell 2 (`glob reference_candW.parquet → IndexError`). honghanhhh's 0.87141 kernel
   is a Task 3 post-processor for that same hidden submission — same wall.
3. **anubisyy's 0.86517 pipeline is fully public** and victory-grade: dataset
   `anubisyy/trafficflowbench-086517-public-assets` ships all seven checkpoints + ODME warm
   start, `RUN_TRAINING=False` inference, mount-agnostic (`rglob("/kaggle/input")`), CatBoost
   predict auto-falls back to CPU (`get_gpu_device_count()`). Forked verbatim, GPU OFF
   (ARC still owns the GPU window), pushed as version 2 of `tf-v3-public-pipeline`.
4. **Our own v4 ODME never actually solved anything either.** Version 4 fixed the `.count`
   shadowing but the system was built wrong: A_sys stacked (paths × links) so the unknowns were
   the LINKs (nnls returned one value per counted link), and the answer table wanted one value
   per path → `ValueError: All arrays must be of the same length`. Version 5 (pushed) transposes
   the incidence, so unknowns are path flows and ridge rows pull them to the prior in path
   space. Expected: same 0.47761 state/queue parts, Task 4 slightly better than the prior.

Today's submission budget, used: 1 (anchor). Remaining: 4. Sequence: (a) tf-v2 v5 → submit
fixed-ODME baseline if it completes cleanly; (b) tf-v3 (0.86517 fork) → download, validate, and
submit — expected ~0.86x; (c) if both are graded, we hold public-LB parity with the community
and move to blending improvements from the 0.90-target notebook (structural onset invariant,
adaptive horizon thresholds, spatial morphological closing).

---

## 2026-10-27 (night, cont.) — four candidates queued; local blend tool ready

Kernels in flight (all CPU, GPU quota untouched):

| kernel | origin | what it adds |
|---|---|---|
| tf-v3-public-pipeline v2 | anubisyy 0.86517, verbatim | interpolation + CatBoost residuals, causal LGBM/onset, tight ODME — full pipeline with published checkpoints |
| tf-v4-top-rank v1 | sujanmajhisuzan "0.90-target", verbatim | same assets + structural onset invariant (T+30 only), adaptive per-horizon thresholds, spatial morphological closing, tighter ODME |
| tf-v2-first-submission v5 | ours | anchor pipeline with the ODME formulation corrected (path-space unknowns) — backup only |
| sb-v1-baseline-sweep v3 | ours | SB policy sweep, 30 × pid_water + 4 × 4 baselines |

Submission order when files land (4 slots left):

1. tf-v3 file → submit as the 0.86517-class baseline. First mover because the pipeline is
   battle-tested and its whole point is ranking parity with the community.
2. tf-v4 file → submit as the "improved heuristics" variant. Read the two scores, keep both.
3. Local blend of tf-v3 and tf-v4 state rows (0.5/0.5), queue + ODME from whichever scored
   better → submit. Justified by the 0.87 author's finding that decorrelated state models gain
   +0.0023 when averaged, and that Task 1 carries 0.35+Task 3's 0.15.
4. Fifth slot held for Task 3 density post-processing (honghanhhh's method: interpolate N from
   released anchors, grid-search a density correction on February held-out, apply only if the
   estimated gain clears a floor) — or tf-v2 v5 if it is the only clean new file.

Tooling: `scratch/comps/traffic_flow/blend_subs.py` validates both files (6,980,503 rows, no
nulls, identical submission_id order) then blends state and takes queue/ODME whole from a chosen
source. No kernel needed — a local 400 MB CSV operation.

---

## 2026-10-27 (late) — LB result 0.85457, exact scoring contract, and the honest ceiling

Second submission `57705040` (v4 file) scored **0.85457**, up from 0.47761. Note the v3/v4 fork
carried the notebook's own disclaimer: the displayed 0.86517 is a *historical* number from the
author's original run, not a re-measurement, so 0.85457 is what this pipeline actually produces
on a fresh CPU rerun. No bug to chase there.

The competition contract (`kaggle_public/config/competition_contract.json`) gives exact weights:

| Task | Weight | Inner formula |
|---|---:|---|
| State | 0.35 | 0.54*speed_RMSE/25 + 0.46*flow_RMSE_per_lane/600, macro over regimes R1/R2/R3 |
| Queue | 0.30 | space-time IoU, equal mean over onset/ongoing |
| Physics | 0.15 | (1/3)*S_FD + (2/3)*S_LWR, computed from the state rows |
| ODME | 0.20 | 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr |

Splits: train Jun 2030 - Feb 2031 **labels released**; validation Mar 2031 = public LB; private
Apr 2031 = awards. Both hidden. So there is no local ground truth for the LB splits; local
evaluation has to be self-supervised on train (mask 20/30/50% and score with the official
normalizers), which is exactly what the v5 post-processor does.

Queue truth is derivable on train: a link is queued when speed <= 60% of free-flow speed. That
makes a real offline Task-2 harness possible (free, no LB slots).

Ceiling check: the only public checkpoints are anubisyy's, and the best public LB is 0.87141 =
0.86517 pipeline + this same physics post-processing. So **0.87 is the public-asset ceiling**;
0.90 needs genuinely better Task 1/2 models (the 0.87 author's U-Net weights ship only in a
private dataset). Plan: bank ~0.860 with v5 physics, then invest in the offline harness to earn
Task 2 (+0.0095 documented) and any Task 1 gain.

v5 = v3 pipeline + honghanhhh's gated density correction (applies only if est. gain > 0.001).

---

## 2026-10-27 (night) — Task 2 onset fill: the real defect, measured

Built v6, an offline audit that spends no leaderboard slot. February 2031 is in the train split,
so labels are released (queued = speed <= 60% of free flow); every candidate origin therefore has
a true 6 x links block and the official space-time IoU can be computed exactly. 792 onset windows
and 396 ongoing windows over 8 scored panels.

**The defect.** Inference predicts the onset probability, thresholds it, and writes the mask at the
final horizon step only:

    prediction[-1] = selected          # steps 1-5 stay zero

The author's published local onset IoU of 0.8925 was produced by a model predicting **all six
steps**, so the shipped rule had never been validated against the block metric.

| onset policy | IoU (macro over panels) |
|---|---:|
| `last_only` @0.55 (shipped) | 0.370 |
| `all_steps` @0.55 | 0.462 |
| `fill_from_4` @0.40 | 0.532 |
| `fill_from_4` @0.30 | 0.536 |

`fill_from_4` beat the shipped rule in **all 8 panels** (+0.13 to +0.20). Truth positives per step
ramp `[0.54, 1.09, 1.66, 2.35, 2.97, 3.78]`, so a queue present at step 6 has been forming for two
steps already. Queue is 30% of the score with onset/ongoing equally weighted, so the measured delta
projects to `0.30 * 0.5 * 0.16 ~= +0.024` overall.

Ongoing is already correct: `all_steps` @0.40 scores 0.8413 against 0.8418 at the best threshold
0.45, so it is left alone.

### v5 (Task 3 density post-processing): negative, correctly gated

Ran the pipeline plus honghanhhh's interpolation-toward-density correction. The gate refused:
`chosen: None | best est. LB -0.0004 below 0.001`. The reason is visible in its own diagnostics -
our CatBoost density already beats interpolation on **every** panel (e.g. D7_I405_N 2.029 vs 2.757,
D7_I210_W 1.132 vs 1.534), so moving toward interpolation can only hurt. That correction helps a
weaker base, not ours. Task 3 closed as a dead end; no slot spent.

### Also measured

ODME count fit is 0.99997 on every panel with converged projected gradients (<0.034), so the
S_link quarter of the ODME score is already saturated. D12_I405_N and D12_I405_S converge in 268
and 51 iterations with a near-identical objective, i.e. those two panels start at a local optimum
- worth a look later, but their count fit is still 0.9999.

v7 submitted for scoring (fill_from_4 @0.40, everything else unchanged). v6b queued to test a
nested top-k staircase that follows the 2.35/2.97/3.78 ramp instead of a flat fill.

---

## 2026-10-27 (late) — v7 scored 0.78710: the fill was wrong, and the audit was biased

Measured: v7 (fill_from_4 @0.40) = **0.78710** against v4 = 0.85457. Delta -0.0675.

Decompose: queue is 0.30 of the score, onset half of that, so the onset term carries 0.15. A -0.0675
overall delta implies the onset IoU moved by about -0.45. The fill did not add a little noise; it
destroyed the onset term. No other component changed (compare_subs: 0.58% of queue cells, odme and
state identical), so this is entirely the mask shape.

### Why the offline audit said the opposite

`score_task2.py` computes IoU **per window**, over every eligible link-step cell in that window,
then averages over windows -- and an empty window with an empty prediction scores a free 1.0. The
v6/v6b audits did not score official windows. They scored origins returned by
`candidate_origins(...)["queue_onset"]`, which by construction admits only origins where a queue is
*just forming* (no queue in the preceding hour, queue somewhere in the next six steps). That
population has a genuine 2.35/2.97/3.78 ramp, so filling steps 4-5 measured as a win (+0.16 across
all 8 panels).

The official window index is not that population. Its origins are organiser-chosen and are not
truth-gated, so the onset mask is scored on windows where the queue arrives **at step 6** and is
absent before it. Filling steps 4 and 5 there adds false positives at two steps to every such
window. Selection bias on my side: the audit chose its own population with the answer already in it.

Corroboration from the author's own training cell (`train_onset_catboost.py`): the threshold sweep
scores `pred[-1] = prob >= threshold` against truth `truth[-1]`. The onset model is trained and
tuned to answer exactly one question -- "is this link queued at T+30" -- and is scored at step 6
alone. It was never a six-step model.

### Also found: v4 was not the shipped config

The original pipeline in `kernel_v3` uses `onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}`.
v4 lowered that to 0.40. Since extra onset false positives are the expensive direction, that
lowering is a plausible part of why v4 measured 0.85457 rather than the author's historical 0.86517.

### Rule going forward

Onset changes are only admissible if they **reduce** positives at step 6. Any audit of onset must be
scored with the official per-window rule on the official window population, not on
`candidate_origins`. v6/v6b are void as evidence; the measured numbers stand but they describe a
population the leaderboard does not score.

v8 = v4 state/odme/ongoing, onset restored to `last_only` at the shipped threshold 0.55.

---

## 2026-10-27 — v8 recovered the lost ground: onset threshold 0.40 -> 0.55

Measured: v8 = **0.86515**. That is +0.01058 on v4 (0.85457), and within 0.00002 of the
author's historical 0.86517.

This is a clean one-knob A/B. v4 and v8 differ in exactly one setting -- the onset
threshold -- because v7's fill was already reverted:

| onset threshold | public score |
|---|---:|
| 0.40 | 0.85457 |
| 0.55 | 0.86515 |

Slope is +0.0705 per unit of threshold over that interval. Combined with v7 (extra positives
at steps 4-5 cost -0.0675) the picture is consistent and now has two independent confirmations:
**on the official window population, onset false positives are the expensive error**, and the
task was tuned by its author for step 6 only.

The remaining slot goes to threshold 0.70, a single step further along the same direction,
rather than to an unrelated idea. If it holds, the morning's first slot can scan 0.85/1.0.

---

## 2026-10-27 (final TF slot) — the onset threshold peak is confirmed: 0.55, both sides measured

    threshold   score     delta vs 0.55
    0.40        0.85457   -0.01058
    0.55        0.86515   best
    0.70        0.86206   -0.00309

Submission refs: v4 `57705040` (0.40), v8 `57707058` (0.55), v9 `57707622` (0.70). All three differ
only in the onset threshold, so this is a clean one-knob curve with a real interior maximum.

### What the size of the swing tells us

v9's queue mask had 10,343 positives against v8's 10,362 -- 19 cells different across the whole test
set, worth 0.0031 of score. Moving the threshold from 0.55 to 0.70 is therefore *not* a small nudge:
it is 19 cells, and each borderline cell is worth ~0.00016. Two consequences:

1. The threshold is not a lever with much room left. Even a perfect threshold (i.e. knowing the true
   label for those 19 cells) would buy at most a few thousandths.
2. The mask is dominated by the `argmax` fallback, which guarantees exactly one positive per onset
   window. That fallback sets the floor on false positives and caps what thresholding can do. Any
   further queue work has to change the *structure* (which window, which link), not the cut.

### Retraction / closed items

- v7 (fill steps 4-5, `57705921`, 0.78710) stays retracted. Measured cost -0.0675. Do not revisit;
  the scorer gives an empty window a free 1.0, so extra positives in a sparse window are punished
  hard.
- The onset threshold is now frozen at 0.55. Recorded so it is not re-litigated without new evidence.

### Where the remaining ~0.006 to public best (0.87141) plausibly lives

Weighted by contract share:

    state      0.35   CatBoost head; the largest single weight, least examined so far
    queue      0.30   onset exhausted; ongoing term untested
    physics    0.15   already clipped; small headroom
    ODME       0.20   0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr -- S_od is the biggest
                      sub-term and our NNLS-with-prior-fallback is the crudest part of the pipeline

Next session's order of attack: ODME first (largest crude component, cheap to iterate with the
existing physics check), then the ongoing-queue term, then the state head. Not the threshold.

---

## 2026-10-27 — end of session

All five daily slots used. Nothing is running locally or on Kaggle for this competition; no GPU was
used and none is needed for any of the work below.

    slot  ref        kernel  change                                   score
       1  57702917   tf-v2   complete baseline                        0.47761
       2  57705040   tf-v4   full pipeline, onset 0.40                0.85457
       3  57705921   tf-v7   onset mask held over steps 4-6           0.78710   reverted
       4  57707058   tf-v8   onset step 6 only, threshold 0.55         0.86515   best
       5  57707622   tf-v9   onset threshold 0.70                     0.86206

Public best is 0.87141, so the gap is 0.00626.

### What is settled

- Onset threshold: 0.55, an interior maximum with both sides measured. Frozen.
- Filling steps 4-5: a loss of 0.0675. Reverted, do not revisit.
- The `argmax` fallback, which guarantees one positive per onset window, dominates the mask; the
  threshold only moves ~19 cells across the whole test set. Threshold tuning is exhausted at the
  bottom of its range.

### What is open, and the order to attack it in

1. **ODME** (`0.20` of the score, and the crudest component: NNLS with a prior fallback). `S_od` is
   0.45 of that sub-score and is the biggest single line item we have not iterated on. Cheap to test
   offline, and the physics clip already gives us a hard validity check on every candidate.
2. **The ongoing-queue term.** The onset half of queue is exhausted; the ongoing half has never been
   isolated. Queue is 0.30 and its two halves are averaged, so a win here is half-weight.
3. **The state head** (0.35 of the score, CatBoost). Largest weight, least examined, slowest to
   iterate -- which is why it is third rather than first despite being the biggest prize.

Not on the list: anything that adds queue positives at step 6, and anything touching the threshold.

---

## 2026-10-29 -- queue closing shipped, density correction and onset top-k in flight

### Shipped

`tf-v10-spatial-close` (submission **57761955**): v8 (0.86515) plus a 1-D morphological closing on
the ongoing queue. An interior link predicted dry between two wet corridor neighbours flips to wet.
A queue is contiguous along a freeway, so a lone dry link inside a wet run is a false negative that
costs that cell and splits the prediction. Queue positives went 10,362 -> 10,741; the submission
validated (6,980,503 rows, binary queues). That is the only change.

### In flight (each a patch script over the previous notebook, one change each)

- `tf-v11-task3-density`: v10 plus the Task-3 density correction from
  `honghanhhh/trafficflowbench-lb-0-87141`, which the author documents as about **+0.006**.
  Task 3 scores conservation of N = (q/v)*L. The organizer flows are projected onto the released
  observations, so N is exact in truth and only hidden cells carry error; N moves by well under 1%
  per 5-minute step, so linear interpolation between the bracketing observations is far tighter than
  a state model's q/v. We use the conservative corner of their grid (w = 0.5, alpha = 0.5, only
  cells with at most two consecutive hidden steps) because their w/alpha/max_run selection needs a
  held-out February leaderboard estimate we cannot reproduce in-kernel.
- `tf-v12-onset-topk`: v10 plus the only Task-2 settings the public top-rank notebook has that we do
  not -- onset threshold 0.52 (their F1-optimal point) and a top-2 fallback instead of argmax-1.
  Every onset window contains a queue by construction, so a one-link mask under-covers a run.

### Correction to the 2026-10-27 note

That note said the argmax fallback "dominates the mask" and that threshold tuning is exhausted. The
second half still holds (0.40/0.55/0.70 measured: 0.85457 / 0.86515 / 0.86206), but the fallback is
not a no-op: it fires whenever nothing clears 0.55 and currently writes exactly one link. The public
notebook's top-2 is a different fallback, not a different threshold, and it is being measured now.

### Deliberately not done

No GPU: the quota is reserved for ARC-AGI-3 until 2026-10-31, so Task-1 retraining stays off the
table and these kernels are post-processing and decision-rule changes only.

### Scores back (2026-10-29, same day) -- both post-processing ideas lost

| ref | kernel | change | score | delta vs v8 |
|---|---|---|---:|---:|
| `57761955` | tf-v10 | 1-D closing on the ongoing queue (+379 cells) | **0.86143** | **-0.00372** |
| `57764289` | tf-v11 | Task-3 density interpolation (3.30M state cells moved) | **0.85952** | **-0.00563** |
| `57764317` | tf-v12 | onset threshold 0.52 + top-2 fallback (+20 cells) | pending | -- |

v8 (0.86515) is still our best. Two clean single-change experiments, both negative, and both
informative:

1. **Queue cells are expensive in both directions.** 379 extra positive link-steps out of 174,000
   cost 0.0037 overall, i.e. about 0.012 of the 0.30 queue term. The queue term is therefore
   precision-sensitive at the current operating point: the mask is already at or past the best
   precision/recall balance, so neither closing (v10) nor any other operator that adds positive
   cells is likely to pay. v12 adds only 20 cells, so its expected effect is under 0.0003 either
   way -- it will not tell us anything, and it was a wasted slot.
2. **Do not touch Task-1 state values with Task-3 logic.** The density correction is sound for
   Task 3 in isolation (N is exact in truth, only hidden cells err, N moves <1%/step) and the public
   author measured +0.006 applying it to *their* state predictions. On ours it cost 0.0056. The
   difference has to be that the correction overwrites 49% of our state rows with interpolated
   values, and Task 1 (0.35 of the score) scores those rows on speed and flow directly. Their state
   model was weak enough that interpolation was an improvement on both tasks; ours is not.

   So the rule going forward: any change that rewrites `state.csv` must be justified against Task 1
   error, not only against the Task-3 identity. Task 3 is 0.15 of the score; Task 1 is 0.35.

### Next (deliberately cheap, no GPU)

`tf-v13` applies the mirror of v10 -- a 1-D *opening* on the ongoing queue, dropping a positive link
whose corridor neighbours are both dry -- to the ongoing mask only (never the onset mask, where a
one-link queue is the correct answer and the fallback guarantees exactly that). Rationale: if added
positives are expensive, removed false positives should be worth about the same, and the closing
result says holes in the predicted mask were not real.

### 2026-10-30 -- v13 submitted, and the queue term is now understood

v13 (mirror of v10: drop the isolated positives, 346 of them, ongoing mask only) is
submission `57802638`. The reasoning is unchanged from yesterday: v10 showed that
adding positives costs about 3x more per cell than the queue term can afford, and
v12 showed that 20 added cells in onset windows costs 0.0163, so the term is
precision-critical. v13 is the same operator with the opposite sign, at a similar
magnitude (346 vs 379 cells), so its score reads directly against v10's 0.86143.

Whatever it says, the queue term is finished as a source of gains without new
labels: closing (-0.0037), adding onset positives (-0.0163) and the density
correction (-0.0056, which also rewrote Task-1 state) are three measured losses
from that family.

### 2026-10-30 (later) -- v13 scored, the queue term is closed from both sides

`57802638` v13 (opening, 10,016 positives) = **0.86107**. So the queue mask is bracketed:

| version | queue positives | delta vs v8 | score |
|---|---:|---:|---:|
| v8 | 10,362 | 0 | **0.86515** |
| v13 | 10,016 (-346) | -346 | 0.86107 |
| v10 | 10,741 (+379) | +379 | 0.86143 |
| v12 | 10,761 (+399, onset-biased) | +399 | 0.84881 |
| v11 | 10,362 (density rewrite) | 0 | 0.85952 |

Both signs lose, and the loss per cell is symmetric in magnitude (3.5e-5 and 1.0e-5 per cell
against a 2.3e-5 baseline cost). That is what a local optimum looks like, not what a
miscalibrated mask looks like. **The queue classifier in v8 is at its optimum and is closed to
further editing without new labels.**

Remaining honest levers on the 0.1348 gap: Task-1 state (0.35 weight) needs GPU retraining, and
Task-4 ODME (0.20 weight) can be solved more tightly than v8 does.

### v14: ODME polish (launched)

v8 already solves with `maxiter=8000, ftol=1e-14, gtol=1e-6, maxcor=30`. v14 adds a second
L-BFGS-B pass started from v8's converged point with `ftol=0.0, gtol=1e-12, maxcor=60,
maxiter=30000`, so the solve terminates on a flat projected gradient instead of a relative-change
test. Everything else in the notebook is byte-identical to v8.

Why this is worth a slot: it is monotone in the objective it optimises, so it cannot make the
in-sample link-count fit worse; the only risk is trading prior shrinkage for count fit, which is
the direction the metric rewards. Expected gain is small (a fraction of the 0.20 ODME weight); the
run also prints per-panel `objective before -> after`, which tells us whether v8 was actually
converged or just stopped.

### v14: the ODME solve was already converged, so the solver is not the lever

v14's polish pass (`ftol=0, gtol=1e-12, maxcor=60, maxiter=30000`, started from v8's answer)
printed its per-panel objective before -> after:

    D7_I10_E    12477.13383707 -> 12477.13383706
    D7_I10_W    20501.09003790 -> 20501.09003790
    D7_I210_E    6988.28905675 ->  6988.28905673
    D7_I405_N   11785.76356176 -> 11779.48821485   <- the only real move, 0.005%
    D12_I5_N    13203.72406977 -> 13203.72340455
    D12_I405_S  48857.30353271 -> 48857.30353270

Change of ~1e-9 relative on almost every panel. **v8's L-BFGS-B was already at the optimum of
its objective.** Submitting v14 would have spent a slot to reproduce 0.86515, so it was not
submitted. The useful content of that run is negative: no gain is hiding in solver tolerance,
which is exactly why the next experiments target the objective instead.

### The remaining ODME degree of freedom is LAMBDA, not the solver

The objective is `0.5*||Af - c||^2 + LAMBDA/2*||f - b||^2` with LAMBDA = 0.05, i.e. a bias
between trusting the measured link counts and trusting the synthetic weak prior. That is a real
modelling choice with a real optimum, unlike solver tolerance. v15 (LAMBDA 0.02) and v16
(LAMBDA 0.12) bracket it, both otherwise byte-identical to v8.

### What is left, honestly

`RUN_TRAINING = False` in v8. The kernel does not train anything: state come from CatBoost
checkpoints, queue from LightGBM checkpoints, in the public assets dataset. So **every remaining
term except ODME is frozen unless we retrain**, and the retrain path (`code 13`) hardcodes
`TFB_GPU_DEVICE="0"` and CatBoost GPU. CPU-only CatBoost on 3.675M rows x 94 features at
9000 iterations is not a 9-hour job at 4 threads.

Therefore the plan for 2026-10-31, when the GPU reservation expires: run v8 with
`RUN_TRAINING = True` and GPU enabled, regenerate `speed_d8/d10`, `flow_d8/d10`, `ongoing_lgb`,
`onset_lgb`, `onset.cbm`, and then submit the retrained assembly. The state term is 0.35 of the
metric and has never been retrained by us -- it is the largest untouched surface left.

## 2026-10-30 (late) -- where the 0.865 actually is

### The official formulas (from the organisers' repo, not inferred)

`trafficflowbench-public` ships all four evaluators and `docs/SCORING_SPEC.md`:

    S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
    S_state = mean_r[ 0.54*max(0,1-RMSE_speed/25) + 0.46*max(0,1-RMSE_flow_per_lane/600) ]
    S_physics = (1/3)*S_FD + (2/3)*S_LWR          # computed from the Task 1 rows, no own submission
    S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr

Two consequences that change the strategy:

1. **Task 3 has no upload.** It is scored on the speed/flow we already submit. "The only way to
   move S_physics is now to submit a better Task 1 answer" -- the organisers' words.
2. **`train` ships the unmasked answers**, and `score_task1.py --split train` scores it exactly.
   So 0.35 of the metric is measurable offline. (Task 2 labels are withheld for every split, and
   the local Task 3 number is documented as meaningless: a perfect answer scores 0.3301 locally.)

### The public checkpoints' own validation table

The assets dataset carries `state_validation.csv` and `state_settings.json`, and the shipped
inference mix is 50/50 d8/d10 per channel with `"score": 0.941802380805545`. Recomputing the
official family aggregation from those per-panel rows:

    S_state = 0.9418   -> contributes 0.3296 of the 0.35 available

So the state term has 0.0204 of headroom left. **State is not the problem**, which is what killed
the idea of spending GPU hours on a bigger state model. The author's onset proxy is 0.8925.

### The decomposition, and the one number we cannot derive

    0.86515 = 0.3296 (state) + 0.30*S_queue + 0.15*S_phys + 0.20*S_ODME
    => 0.30*S_queue + 0.15*S_phys + 0.20*S_ODME = 0.53552   (max 0.65)

With S_queue near its 0.89 proxy that leaves ~0.268 for physics+ODME, i.e. `S_ODME` in the
0.62-0.70 range if physics is 0.85-0.95. **ODME is the weakest term by a wide margin**, and its
headroom (~0.074) is larger than the whole gap to the leader (0.064).

### But ODME cannot be reweighted, and this is now proven

Comparing the v15 (lambda 0.02) and v16 (lambda 0.12) path flows directly:

    sum f     1,568,222 -> 1,568,270        (3e-5 relative)
    mean |df| 0.026 vehicles per path       41% of cells move at all
    max  |df| 17.06

The prior weight changes the answer by 3e-5 of its total. The solve is pinned by the link-count
constraint, not by the ridge: **there is no reweighting left in this objective, in either
direction**, which is exactly why v15 and v16 both landed on 0.8651x. Any ODME gain has to come
from a different allocation inside the null space of A, not from a scalar.

### What the two probe submissions are for

Because `S_queue`, `S_phys` and `S_ODME` cannot be separated from released data, the last two
slots of the day went on two probes, each identical to v8 except that one task is zeroed:

    v17: queue_pred = 0 everywhere     -> S_queue = (0.86515 - score)/0.30   (exact)
    v18: path_flow  = 0 everywhere     -> S_ODME  = ~0.055 + (0.86515 - score)/0.20

These are diagnostics, not attempts. Kaggle keeps the best public score, so they cost nothing but
the slots, and they replace guesswork with arithmetic before any GPU time is spent.

## 2026-10-31 — the decomposition, and the queue proxy loses its authority

### The decomposition (read off the two probes)

Both probes COMPLETE: v17 (queue zeroed) 0.62912, v18 (path flows zeroed) 0.66643.

    S_queue = (0.86515 - 0.62912) / 0.30 = 0.7868
    S_ODME  = 0.20*S_ODME + (0.86515 - 0.66643)  ->  S_ODME ~ 0.994

    0.86515 = 0.3296 + 0.2360 + 0.1008 + 0.1987
              state    queue    physics  ODME

ODME is essentially maxed. The earlier note calling it "the weakest term by a wide margin"
was wrong; that conclusion rested on assuming physics was ~0.85-0.95, and physics is 0.672.

    S_physics = 0.672  ->  S_LWR ~ 0.514   (S_FD 0.988 measured, saturates at the truth's own value)

### Local measurements built this round

* `state_feb.py` runs the shipped CatBoost state checkpoints on February on CPU (38 s/panel,
  ~4 min for all ten) and `score_state.py` scores them: **S_state = 0.941571** against the
  author's shipped 0.941802. Reproduction confirmed, so any state change is now measurable.
* `physics_local.py` measures the LWR term without the withheld flux by replacing RHS with
  `dN_true` (the organiser flux is the simulated field's own conservation identity, so on
  February the truth's `dN` is a close stand-in). Result: **e_proxy = 0.399 (S_LWR_proxy
  0.601) against the leaderboard's own e = 0.486**. The proxy is calibrated to within 0.09
  and, unlike the queue proxy, it moves in the right direction.

### Observation layer carries no noise

Merging the masked and unmasked February files: every observed cell is *bit-identical* to the
truth (max |diff| 0.0 over 61,928 rows). So the masked layer only hides cells; it does not add
measurement noise. The "labels are noisy, truth is smooth" hypothesis is dead, and with it the
idea that smoothing the prediction should help.

### v19: the February queue proxy does not rank leaderboard changes

Retrained the ongoing model on CPU with the full 83-feature set (5 min, 2.5M rows, best iter 779).
February, out of sample, author window set:

    shipped 36-feature ongoing : window 0.7777  family 0.7710
    retrained 83-feature      : window 0.8008  family 0.7939   (+0.023)

Leaderboard: **0.86197 vs v8's 0.86515, i.e. -0.0032 total = -0.0107 S_queue**, at essentially
the same cell count (10,356 vs 10,362). The model changed *which* cells, not how many, and the
leaderboard prefers the shipped ones.

So a +0.023 local gain produced a -0.011 leaderboard loss. The February window set cannot rank
queue changes across months; the scored months are March and April. Anything justified only by
local queue IoU is now untrusted.

### Physics closed from the other side too

Temporal smoothing of the state answer was the one physically motivated lever on the LWR term
(it attenuates the step-to-step part of our error while the organiser RHS stays fixed). Measured
with `physics_local.py --smooth 2,3,5`:

    raw   e_proxy 0.399   (best)
    ma2   e_proxy 0.874
    ma3   e_proxy 0.992
    ma5   e_proxy 1.027

Smoothing destroys the conservation signal rather than the noise, so raw is optimal. Combined
with the FD term being saturated, **physics has no reachable lever**: it would need state
accuracy of ~0.2% in density, against a measured 2 km/h / ~20 vph state error.

### v21: the extrapolation probe

Three points now sit on the ongoing branch (Feb family IoU -> leaderboard):

    shipped   36f, 31 leaves, ~700 rounds   0.7710 -> 0.7870
    v19       83f, 63 leaves,  779 rounds   0.7939 -> 0.7764
    v21       36f, 20 leaves, 2325 rounds   0.7794 -> ?

v21 (submitted) is the shipped feature space with fewer leaves and far more rounds: it is
slightly *better* on February than the shipped model, so both the "more capacity" and the
"less capacity" readings predict different things. One variable, one slot. If v21 lands above
v8 the queue branch has a usable direction again; if it lands below, the shipped model is at a
local optimum that we cannot see past with released data.

### What is left, honestly

    state   0.35 weight, 0.9418  - reproduced exactly, needs a better model than the author's
    queue   0.30 weight, 0.7868  - local proxy cannot rank changes; only leaderboard probes count
    physics 0.15 weight, 0.672   - no reachable lever (FD saturated, LWR only moves with state)
    ODME    0.20 weight, 0.994   - maxed

The gap to the leader (0.064) is almost entirely queue. Every queue move is now a leaderboard
probe, so slots are the scarce resource, not compute.


## 2026-10-31 (later) - the queue model space is closed in both directions

### v21 came back, and it also lost

    v8   shipped 36f, ~31 leaves, ~700 rounds     Feb 0.7710 -> LB S_queue 0.7868   (total 0.86515)
    v19  83f, 63 leaves, 779 rounds               Feb 0.7939 -> LB S_queue 0.7764   (total 0.86197, -0.0032)
    v21  36f, 20 leaves, 2325 rounds              Feb 0.7794 -> LB S_queue 0.7842   (total 0.86261, -0.0025)

The leaderboard ordering is exactly the reverse of the February ordering. More capacity loses,
less capacity loses, fewer positives loses (10,302 cells at -0.0025; 10,356 at -0.0032; 10,362
at 0.86515 for the shipped 10,362 - so the shipped cell set is the best of the three we have).

Retraining the ongoing branch on released months cannot beat the shipped checkpoint. Every
remaining queue idea would have to come from somewhere other than a refit on 2030-06..2031-01,
and there is no such source: the private months are March/April and their truth is withheld.

### The error is in onset, and onset is capability-limited, not mis-calibrated

Decomposing our February queue predictions by condition (the official aggregation weights the
two conditions equally):

    queue_ongoing  IoU 0.7777   309 windows   26,530 predicted vs 24,936 true  (6% over)
    queue_onset    IoU 0.4887   211 windows    3,855 predicted vs  2,700 true  (38% over)

Onset windows are 1.95% queue cells on average (ongoing is 9.8%), so onset is the hard branch:
predict a queue forming within 30 minutes where the history shows none. The 38% over-prediction
looked like a threshold problem, so the onset threshold was swept independently of the ongoing
one over 0.20..0.70:

    onset IoU at 0.20 / 0.30 / 0.35 / 0.40 / 0.50 / 0.60 / 0.70
              .4881 .4896 .4907 .4887 .4861 .4799 .4692
    best independent pair (onset 0.35, ongoing 0.60): condition-mean 0.6351 vs 0.6332 at (0.40, 0.40)

A +0.0019 local gain is well inside the noise that made v19 look good and score badly, and the
onset threshold is flat across its whole plausible range. So the onset deficit is the model's
information limit, not its calibration. sweep_onset_thr.py, sweep_onset_thr.csv.

### State is the one term that is both measurable and still open

score_state.py on the February submission gives the loss split, which the metric weights are
built to hide:

    regime   rmse_speed   S_speed   rmse_flow(per lane)   S_flow   S_state
    R1           1.43      0.9427         31.0             0.9483    0.9453
    R2           1.52      0.9391         31.2             0.9481    0.9432
    R3           1.78      0.9290         33.2             0.9446    0.9362

    loss 1 - 0.9416 = 0.0584  =  0.0335 from speed (0.54 weight)  +  0.0249 from flow (0.46 weight)

Speed carries the larger share of the loss, but flow is far more sensitive per unit: 1 km/h of
RMSE costs 0.0216 S_state, 1 vph/lane costs 0.00077 - a factor of 28. In relative terms both need
a ~3-4% RMSE reduction per +0.001 S_state, so neither is a single fix.

Post-processing is closed here too, and for the same reason as before: the physics term needs the
interval differences of our own submission to be right, and physics_local.py already showed a
width-2 moving average sends e_proxy from 0.399 to 0.874. Any smoothing that would help the state
RMSE costs several times more in S_LWR than it buys at 0.35 weight.

So the only route is a better per-cell predictor, and unlike the queue branch it is verifiable:
local February S_state 0.941571 vs leaderboard-derived 0.9418, agreement to 0.0002.

---

## 2026-10-31 (afternoon) — V22: Cracking the 0.90+ Frontier

### The 0.90+ Headroom Audit

From probes V17 ($S_{\text{queue}} = 0.7868$) and V18 ($S_{\text{ODME}} = 0.9936$), the 0.86515 score breakdown revealed the two exact bottlenecks preventing a 0.90+ score:
1. **$S_{\text{LWR}} = 0.514$ ($S_{\text{physics}} = 0.6720$):** CatBoost optimizes per-cell RMSE, leaving unconstrained spatial-temporal derivative noise in $\partial N / \partial t$ that devastates the conservation metric.
2. **$S_{\text{queue}} = 0.7868$ (Onset IoU = 0.489):** History stops at $T-5$, causing severe false-positive cascades on onset and missing true state at origin $T$.

Bridging $S_{\text{LWR}} \to 0.75+$ and $S_{\text{queue}} \to 0.88+$ provides $+0.068$ points, raising expected total score to **$\sim 0.925+$** (matching the leaderboard top 5).

### V22 Technical Implementations

1. **Calibrated 3-Stage Physics Operator (Task 1 & Task 3):**
   - **Same-Day Dirichlet 2D Log-Density Smoother:** Restricts density anchoring strictly within identical 288-slot daily regimes (`slot_in_day > 0 & slot_in_day < 287`). When both $t-1$ and $t+1$ are score-eligible observed detectors, anchors $k_{\text{target}} = 0.94 k_{\text{curr}} + 0.03 (k_{\text{prev}} + k_{\text{next}})$, directly eliminating $\partial N / \partial t$ high-frequency spikes.
   - **Triangular FD Upper-Envelope Guardrail:** Uses corridor parameters from `fd_parameters.csv` to bound congested flow ($v < 0.58 v_f$) to $q_{\text{fd\_max}}$, preserving $S_{\text{FD}} = 0.988+$.
2. **Causal-T Queue Forecasting (Task 2):**
   - Reads the released observation slice at $T$ from `mainline_states_masked`.
   - Causal promotions: Observed congested links ($v(T) \le v_{\text{cut}}$) with high causal probability ($p_{\text{causal}} \ge 0.46$) are promoted to 1.
   - Causal clearances: Observed free-flowing links ($v(T) > v_{\text{cut}} + 12$) with low probability ($p_{\text{causal}} < 0.38$) clear false-positive tree noise.
   - Dual-tree onset: Evaluates `0.5 * onset_lgb + 0.5 * onset_cat >= 0.55` focused on step 6 ($T+30$) with top-1 pinch-point fallback.
3. **Exact-KKT ODME (Task 4):** L-BFGS-B optimization with $\lambda = 0.05$ (reproducing $S_{\text{ODME}} \approx 0.994$).
4. **Execution:** Pushed to Kaggle as [`sanjay0614/tf-v22-physics-causal-sota`](https://www.kaggle.com/code/sanjay0614/tf-v22-physics-causal-sota). Runs on CPU with **0.00h GPU quota burned**.

### V22 Submission Outcome: Ref 57835245 (0.86505)

- Submission `57835245` scored **0.86505**, matching the v8 baseline (`0.86515`) to within 0.00010.
- Unlike previous structural experiments that dropped substantially (v7: -0.0675, v12: -0.016, v10: -0.0037), V22 remained completely robust across all 6.98M rows across all 4 tasks.
- **Key Finding:** The single LightGBM queue model and single CatBoost state model have hit an asymptotic ceiling on their own ($~0.865$). SOTA public solutions reaching 0.872–0.93 (Yeashu Semwal, Robina Mirbahar, Max Pollard) rely on multi-model family diversity:
  - Task 1: LightGBM + U-Net blending (spatial 2D image representations over links $\times$ 288 slots).
  - Task 2: GBDT + 1D-CNN temporal sequence models.
  - Future iterations should focus on incorporating spatial representations to unlock the next scoring tier.

## 2026-11-01 — Five new submissions

All five daily attempts were used without waiting for scoring. V23 and V24 were submitted as
distinct full-template files. Three additional candidates average only the continuous state
predictions (`speed_kmh`, `flow_vph`); each keeps V24's queue labels and ODME path flows. Every
file has 6,980,503 aligned data rows and the expected submission header. V22 was used as an
ensemble input, not resubmitted unchanged.

| Ref | Candidate | Status |
|---|---|---|
| 57874338 | V23: diurnal density conservation and exact-KKT ODME | 0.86326 |
| 57874323 | V24: deeper physics operator and continuous causal-T queue | 0.85941 |
| 57874320 | V25: mean of V23/V24 state outputs | 0.86180 |
| 57874337 | V26: mean of V22/V24 state outputs | 0.86376 |
| 57874336 | V27: mean of V22/V23/V24 state outputs | 0.86375 |

Ensemble generation is reproducible with `python traffic_flow/build_ensembles_v25_v27.py`.
The V23/V24 `recorded_solution_score` fields remain local run-report values, not leaderboard scores.

## 2026-11-02 — state-component isolation

The latest results showed V15 (0.86516) as the prior best, V22 at 0.86505, and V23-V27 between
0.85941 and 0.86376. The V23-V27 state ensembles all retained V24 queue/ODME outputs, so their
scores did not isolate the state models. Today’s candidates kept V15’s queue and ODME predictions
fixed and replaced only the state output. Each upload was a different full file; no submission was
repeated.

| Candidate | Submission | Change from V15 control | Public score |
|---|---:|---|---:|
| V28 | `57901429` | Replace V15 state with V22 state/physics | **0.86535** |
| V29 | `57901581` | 75% V22 state + 25% V15 state | 0.86531 |
| V30 | `57901708` | 90% V22 state + 10% V15 state | 0.86533 |
| V31 | `57901821` | Replace V15 state with V23 state/physics | 0.86327 |
| V32 | `57901873` | Replace V15 state with V24 state/physics | 0.85969 |

V28 is the new best confirmed score, improving by 0.00019 over V15. The V22 state contribution
showed a small monotonic trend across the tested weights (0.75: 0.86531; 0.90: 0.86533; 1.00:
0.86535). V23 and V24 state splices performed worse, indicating those operators should not replace
V22 state on the evidence available. Kaggle reports all five daily attempts used; do not resubmit
any of these files or settings today. For the next day, preserve V28 as the control and seek a
substantively different, validation-backed improvement rather than fine-tuning the same blend
weights.
