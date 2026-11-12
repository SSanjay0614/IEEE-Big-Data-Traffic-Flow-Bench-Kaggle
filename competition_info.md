# Competition info — 2026 IEEE Big Data Cup, Traffic Flow Bench

Read from the released package and the host's scoring repository on 2026-10-27.

## Identity

| Field | Value |
|---|---|
| Kaggle ref | `2026-ieee-big-data-traffic-flow-bench` |
| URL | https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench |
| Host | 2026 IEEE Big Data Cup (academic benchmark, released with code and evaluators) |
| Category | Community |
| Teams at time of writing | 637 |
| Prize pool | $3,500 |
| Submission type | CSV upload (not a code competition) |
| Submissions per day | 5 |
| Our state | Joined, no submission yet |
| Deadline | **2026-11-07** — 11 days from 2026-10-27 |

## Scoring

```
S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
```

All components lie in [0, 1]; 1 is perfect. Corridors are averaged with equal weight, so a small
panel counts as much as a large one. A missing task output scores **0**, not "excluded".

### Task 1 — state reconstruction (weight 0.35)

Recover `speed_kmh` and `flow_vph` at masked eligible cells.

```
S_speed = max(0, 1 - RMSE_speed / 25)          # km/h
S_flow  = max(0, 1 - RMSE_flow_per_lane / 600) # veh/h/lane
S_state = 0.54*S_speed + 0.46*S_flow
```

Averaged per masking regime, then over regimes (macro mean). Flow error is computed **per lane**:
submitted and true flow are each divided by the station's lane count from `fd_parameters.csv`
before the RMSE.

Masking regimes, assigned one per calendar day:

| Regime | Fraction of eligible cells removed |
|---|---|
| R1 | 20 % |
| R2 | 30 % |
| R3 | 50 % |

At a blanked cell every measured channel is removed (speed, flow, occupancy and the density
derived from occupancy) because they come from the same detector. The quality fields survive.

**Do not derive the mask by looking for nulls.** Around each Task 2 window the horizon and the
following hour are blanked on purpose and are not Task 1 targets. Use
`task1/<PANEL>/<split>/sample_submission_state.csv`, which is the full and only list.

### Task 2 — queue propagation (weight 0.30)

From 60 minutes of history before an origin `T`, predict binary queue status for every mainline
link at `T+5 … T+30` (six 5-minute steps). Two conditions, five windows each per panel and split:

- `queue_onset` — nothing queued in the history, a queue appears in the horizon;
- `queue_ongoing` — a queue is already visible at `T` and must be carried forward.

Both conditions guarantee a queue somewhere in the horizon, so "predict nothing" cannot win.

```
IoU = |predicted AND true| / |predicted OR true|     # per window, space-time
```

Equal-weighted across windows and conditions. A link counts as queued when speed is at or below
**0.60 x free-flow speed**, with the label taken from the true state rather than the noisy
observation.

`D12_I405_N` and `D12_I405_S` are excluded from Task 2 only. They still count in tasks 1, 3, 4.

Published baseline: persistence (repeat the queue state at `T` through all six steps) scores
**0.2518**. Task 2 cannot be scored locally — queue labels are withheld for every split.

### Task 3 — physics consistency (weight 0.15)

No submission. The evaluator reads the Task 1 file and checks two things.

```
S_physics = (1/3)*S_FD + (2/3)*S_LWR
```

- `S_FD` — fundamental-diagram validity. Speed, flow and density must be able to coexist on a real
  road. This is a validity gate, which is why it is the smaller weight. A panel/regime whose
  submitted cells are more than a fifth below 50 vph scores `S_FD = 0`.
- `S_LWR` — conservation:

```
N(t+dt) - N(t) = dt * (q_in + r_on - q_out - r_off)
```

Density is `q/v`, vehicles on a link is density times length, boundary flows come from the
released topology and ramp flows from the released ramp observations.

**Cannot be scored locally.** The evaluator is published, but conservation needs the true
boundary flows, which is the Task 1 answer. The local fallback estimates boundary flows from your
own submitted flows; the conservation signal is about 0.75 % of the vehicles on a link and the
estimate is off by about 0.22 %, so roughly a third of the signal is noise and `S_LWR` reads 0 for
everyone. Reference numbers from the host, on `D12_I5_N`: a perfect answer and the naive baseline
score 0.9598 against 0.3242 when scored properly, but 0.3301 against 0.3299 locally. That is why
we develop against Task 1 and let the leaderboard judge physics.

The gap between those two numbers is the single clearest signal in this benchmark: **cell-wise
accuracy does not imply physical consistency.** A smooth per-cell interpolator makes vehicles
appear and vanish between detectors, and RMSE cannot see it.

### Task 4 — ODME and path flows (weight 0.20)

Given the path set, the path-link incidence matrix `A`, observed link counts `c`, and a weak path
prior `b`, submit one non-negative flow per path.

```
S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr
```

- `S_od` — distance from the reference path flows;
- `S_link` — whether the submission reproduces the observed counts once loaded onto the network;
- `S_dev` — how far you moved from the prior, relative to how far the reference moved. This is the
  term that kills both degenerate answers: submit the prior unchanged, or fit the counts exactly
  with a wild pattern;
- `S_attr` — whether the destination-attraction distribution is right.

Baseline: non-negative least squares with prior regularisation.

## Data

**9,688 files, 12.4 GB unpacked.** Ten directional panels:

```
D7_I10_E   D7_I10_W     D7_I210_E  D7_I210_W    D7_I405_N  D7_I405_S
D12_I5_N   D12_I5_S     D12_I405_N D12_I405_S
```

Per panel:

```
network/   links.csv, fd_parameters.csv, lwr_mainline_topology.csv, ramp_attachment_map.csv,
           path_set.csv, path_link_incidence.csv, base_od.csv
train/     mainline_states/, mainline_states_masked/, ramp_states/
validation/  private/
           mainline_states_masked/, ramp_states/
```

Splits:

| Split | Dates | Days | Labels |
|---|---|---|---|
| `train` | 2030-06-01 → 2031-02-28 | 273 | released, so self-scorable |
| `validation` | 2031-03-01 → 2031-03-31 | 31 | hidden, public leaderboard |
| `private` | 2031-04-01 → 2031-04-30 | 30 | hidden, decides the ranking |

Validation carries five incidents and private seven, generated independently. A method tuned
until it fits validation has learned that month's incidents, and the final ranking will move.

Task directories hold the per-corridor templates: `task1/` state submissions, `task2/`
`window_index.csv` + `window_history.parquet` + queue templates, `task4/` link counts, weak prior
and path-flow templates.

## Quality rule

Every mainline and ramp cell carries `pct_observed`.

```
is_imputed         = pct_observed < 100
is_score_eligible  = pct_observed >= 75 and required values present
```

Only eligible cells are scored, in every task. About **76 %** qualify. Ineligible cells are still
released. An unavailable ramp reading is missing evidence, not a measured zero — treating it as
zero costs Task 3 score.

## Submission file

```
submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
1,state,0,0,0,0
```

- `task` is one of `state`, `queue`, `odme`; fill only the columns that task asks for.
- **6,980,503 rows.** The template is 152 MB. `submission_key.csv` (643 MB, download only) maps
  every `submission_id` to task, split, panel, timestamp, station, link, mask regime, window, path
  and zones. Never upload the key.
- No empty cells anywhere. A blank value rejects the whole submission before scoring.
- There is no `physics` task in the file; Task 3 is scored from the `state` rows.

## Rules worth repeating

- Do not submit values taken from an organiser-only file. Every split has a legitimate route to an
  answer.
- The records are synthetic and must not be presented as real freeway observations.
- Check the competition page for eligibility and prize terms before treating a placement as
  payable.
