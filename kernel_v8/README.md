# TF V8 — onset reverted to step 6 at the shipped threshold

## Why this exists

v7 held the onset mask over horizon steps 4–6 and measured **0.78710**, against v4's
**0.85457**. That is a −0.0675 swing from a change that touched nothing but the onset
mask (verified: 0.58% of queue cells differed, state/ODME byte-identical).

## What actually happened

`score_task2.py` scores **per window**: IoU over every eligible link-step cell in the
window, averaged over windows, with an empty window and an empty prediction scoring a
free 1.0.

The v6/v6b offline audits did not score official windows. They scored origins from
`candidate_origins(...)["queue_onset"]`, which by construction keeps only origins where a
queue is *just forming* — a population with a real 2.35 / 2.97 / 3.78 link ramp across
steps 4–6. Filling steps 4–5 measured as +0.16 IoU there.

The official window index is not that population. Its queue arrives at step 6, so filling
steps 4–5 adds false positives at two extra steps in every such window. The audit picked
its own population with the answer already baked in.

Corroborated by the author's own `train_onset_catboost.py`: the threshold sweep scores
`pred[-1] = prob >= threshold` against truth `truth[-1]`. The onset model was trained and
tuned for step 6 only.

## What v8 changes

Two lines, both reverts:

```python
onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}   # was threshold 0.4
prediction[-1] = selected                                  # was prediction[3:] = selected
```

`0.55` is the value in `kernel_v3`, the 0.86517 configuration. v4 had lowered it to 0.40,
and since extra onset positives are the expensive direction, that lowering is a plausible
part of why v4 measured 0.85457 instead of the author's historical 0.86517.

State model, ODME solve, ongoing threshold 0.40 and the physics clip are unchanged from
v4, so the leaderboard delta isolates the onset term.
