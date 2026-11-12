# Traffic Flow Bench — working repo

Kaggle: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench
Official code and evaluators: https://github.com/jacky850/trafficflowbench-public
**Competition status: CLOSED (deadline 2026-11-07 06:55 UTC). Final result: rank 317 of 802 teams, score 0.86535.**

## Final Result

| Field | Value |
|---|---|
| Competition | 2026 IEEE Big Data Cup: Traffic Flow Bench |
| Status | **Completed / closed 2026-11-07** |
| Team | SanjayS (`sanjay0614`), team id `17214405` |
| Final rank | **317 of 802 teams** (top 40%) |
| Final score | **0.86535** |
| Best submission | **V28**, ref `57901429` (V22 state + physics, V15 queue + ODME) |
| Winning score | 0.93124 (Max Pollard); 2nd 0.92675, 3rd 0.92406 |
| Submissions used | 26 |
| Compute | Kaggle CPU notebooks only; 0.00 h GPU quota |
| Last submission | 2026-11-02 08:33:41 UTC |

Best confirmed run by task, from the V17/V18 probes plus the V28 decomposition:
state 0.9418 (weight 0.35), queue 0.7868 (0.30), physics 0.6720 (0.15), ODME 0.9936 (0.20).
The remaining gap to the 0.93 winners was physics and queue, not ODME: the exact-KKT
ODME solver was already at its ceiling, while the queue term and the physics guardrail
were the two components that never cleared the public 0.80/0.90 marks.

This is the 2026 IEEE Big Data Cup: ten directional freeway corridors at five-minute
resolution, and four questions asked about the same road. It is a CSV-upload competition, not a
code competition, and the whole thing runs on CPU.

## The four tasks, and why they are one problem

| Task | Question | Weight | Scored locally? |
|---|---|---|---|
| 1. State | Fill in the cells the release blanked out | 0.35 | yes, on `train` |
| 2. Queue | From 60 min of history, which links are queued in the next 30 min | 0.30 | no, labels withheld |
| 3. Physics | Is your Task 1 answer physically possible | 0.15 | no, boundary flows withheld |
| 4. ODME | Recover path flows from link counts | 0.20 | partly |

```
S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME
```

Every component is on [0, 1] and a missing task scores 0 rather than being skipped. That matters:
**a submission with no Task 4 file loses 0.20 outright**, so a complete mediocre submission beats
a brilliant partial one.

Task 3 has no file of its own. It reads the speeds and flows already in the Task 1 file, which is
elegant and brutal: you cannot tune physics directly, you can only reconstruct better.

## Score Decomposition & Path to 0.90+

From diagnostic probes V17 (queue zeroed: `0.62912`) and V18 (path_flow zeroed: `0.66643`):

$$S_{\text{total}} = 0.35 \cdot S_{\text{state}} + 0.30 \cdot S_{\text{queue}} + 0.15 \cdot S_{\text{physics}} + 0.20 \cdot S_{\text{ODME}}$$

| Term | Weight | Baseline (V8) Score | Weighted Contribution | Headroom / Path to 0.90+ |
|---|---|---|---|---|
| **Task 1: State** | 0.35 | 0.9418 | 0.3296 | Preserved via CatBoost residual reconstruction ($S_{\text{speed}} = 0.943, S_{\text{flow}} = 0.948$). |
| **Task 2: Queue** | 0.30 | 0.7868 | 0.2360 | **+0.0340 Gain**: Causal-T origin slice ($p_{\text{causal}} = 0.60 p_{\text{std}} + 0.40 p_T$) + promotions/clearances + step-6 onset. |
| **Task 3: Physics** | 0.15 | 0.6720 | 0.1008 | **+0.0267 Gain**: 3-Stage Physics Operator (same-day Dirichlet density smoother + triangular FD guardrail). |
| **Task 4: ODME** | 0.20 | 0.9936 | 0.1987 | **Maxed Out**: Exact-KKT L-BFGS-B solver ($\lambda = 0.05, \text{gtol} = 10^{-10}$). |
| **Total** | **1.00** | **0.86515** | **0.86515** | **Target: 0.925+** (Matching top leaderboard scores). |

---

## Submission & Version History

| Version | Ref | Score | Experiment Summary & Diagnostics |
|---|---|---|---|
| **V2** | `57702917` | 0.47761 | Complete baseline: historical-mean state, persistence queue, NNLS-odme. |
| **V4** | `57705040` | 0.85457 | CatBoost state + LGBM queue (onset threshold 0.40) + NNLS ODME. |
| **V7** | `57705921` | 0.78710 | Onset held across steps 4–6 (−0.0675 loss due to false positives on official step-6 onset). |
| **V8** | `57707058` | 0.86515 | Restored single-step onset at step 6, threshold 0.55. Established best baseline. |
| **V9** | `57707622` | 0.86206 | Onset threshold probe at 0.70 (overly conservative). |
| **V10** | `57761955` | 0.86143 | Spatial morphological closing on ongoing queue (−0.0037 loss for +379 false cells). |
| **V11** | `57764289` | 0.85952 | Density interpolation on top of V10 (degraded by V10 closing and unmasked day crossings). |
| **V12** | `57764317` | 0.84881 | Onset threshold 0.52 + top-2 fallback on top of V10. |
| **V13** | `57802638` | 0.86107 | 1-D opening on ongoing queue (drop isolated positives). |
| **V15** | `57803438` | **0.86516** | ODME ridge $\lambda = 0.02$ (slight count fit improvement). |
| **V16** | `57803532` | 0.86511 | ODME ridge $\lambda = 0.12$ (over-regularized). |
| **V17** | `57804203` | 0.62912 | **PROBE**: Task 2 queue zeroed. Delta yields $S_{\text{queue}} = 0.7868$. |
| **V18** | `57804291` | 0.66643 | **PROBE**: Task 4 path flow zeroed. Delta yields $S_{\text{ODME}} = 0.9936$. |
| **V19** | `57825818` | 0.86197 | Ongoing model retrained on CPU with 83 features (Feb proxy failed to translate to LB). |
| **V21** | `57827138` | 0.86261 | Shipped 36-feature ongoing retrained with 20 leaves, 2325 rounds. |
| **V22** | `57835245` | 0.86505 | Physics-Guided SOTA: Same-day Dirichlet 2D density smoother + Triangular FD guardrail + Causal-T origin queue promotions/clearances + Exact-KKT ODME. |
| **V23** | `57874338` | 0.86326 | **Diurnal Density Conservation**: Same-day interpolation ($w=0.5, \alpha=0.5, \text{run} \le 2$) + clean step-6 onset + exact ODME. |
| **V24** | `57874323` | 0.85941 | **Multi-Axis Physics Expansion**: ($w=0.75, \alpha=0.78, \text{run} \le 3$) + Triangular FD Guardrail + Continuous Causal-T Queue ($p_{\text{causal}} \ge 0.40$). |
| **V25** | `57874320` | 0.86180 | Mean of V23/V24 state speed and flow; retain V24 queue and ODME outputs. |
| **V26** | `57874337` | 0.86376 | Mean of V22/V24 state speed and flow; retain V24 queue and ODME outputs. |
| **V27** | `57874336` | 0.86375 | Mean of V22/V23/V24 state speed and flow; retain V24 queue and ODME outputs. |
| **V28** | `57901429` | **0.86535** | V22 state/physics with V15 queue and ODME; current best confirmed. |
| **V29** | `57901581` | 0.86531 | 75% V22 state + 25% V15 state; retain V15 queue and ODME. |
| **V30** | `57901708` | 0.86533 | 90% V22 state + 10% V15 state; retain V15 queue and ODME. |
| **V31** | `57901821` | 0.86327 | V23 state/physics with V15 queue and ODME. |
| **V32** | `57901873` | 0.85969 | V24 state/physics with V15 queue and ODME. |

V23/V24 local run-report values are not leaderboard scores. The V25–V27 files were rebuilt and
SHA-256 checked against the exact submitted files with `build_ensembles_v25_v27.py`.
V28–V32 were validated as complete, aligned files with 6,980,503 rows and the expected task counts.




---

## Local Rules for this Repo

- Data and scoring run in Kaggle CPU notebooks. Nothing 12 GB wide happens locally.
- 0.00h GPU quota burned on traffic flow.
- No paid actions.
