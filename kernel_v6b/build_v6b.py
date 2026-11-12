"""Offline experiment 2: shape of the onset mask across the horizon (no LB slot).

v6 established that holding the step-6 mask over steps 4-6 beats a single frame
(0.536 vs 0.370 IoU). But truth is not flat across those steps either:

    mean queued links at steps 1..6 = [0.54, 1.09, 1.66, 2.35, 2.97, 3.78]

so a flat fill under-predicts nothing at step 6 while over-predicting steps 4 and 5.
A nested (top-k) staircase with k proportional to that ramp should fit better.

Policies tested on labelled February windows, same windows and truth as v6:
  flat           step-6 thresholded mask repeated over steps 4-6   (v7's rule)
  staircase      nested top-k: k(step 4) < k(step 5) < k(step 6)
  all_steps      threshold the whole block
  last_only      the originally shipped rule (control)

Thresholds are swept widely, including below v6's floor of 0.30.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "scratch" / "kernel_out" / "tf3pub" / "tfb_source"
MODULES = ["config.py", "queue_features.py", "queue_ongoing.py"]

cells = []

cells.append(("markdown", """# Task 2: onset mask shape (offline, labelled February)

v6 measured the **fill policy** and found holding the step-6 mask over steps 4-6 beats a single
frame by +0.16 IoU on every panel. But truth itself ramps across those steps:

```
mean queued links, steps 1..6 = [0.54, 1.09, 1.66, 2.35, 2.97, 3.78]
```

A flat fill matches step 6 and over-predicts steps 4-5. This measures a nested top-k
**staircase** whose width follows that ramp, against the flat fill and the shipped rule, on the
same labelled windows and the same official block IoU."""))

cells.append(("code", """\
import os, sys, glob, json, time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
elapsed = lambda: f"{time.time() - T0:.0f}s"
DATA = Path(sorted(glob.glob("/kaggle/input/**/kaggle_public", recursive=True))[0])
ASSETS = Path(sorted(glob.glob("/kaggle/input/**/onset_lgb.txt", recursive=True))[0]).parent
WORK = Path("/kaggle/working")
(WORK / "tfb_source").mkdir(parents=True, exist_ok=True)
os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(WORK), TFB_THREADS="4")
sys.path.insert(0, str(WORK / "tfb_source"))
print("data:", DATA, "| assets:", ASSETS)"""))

for name in MODULES:
    cells.append(("code", f"%%writefile tfb_source/{name}\n" + (SRC / name).read_text(encoding="utf-8")))

cells.append(("code", """\
import lightgbm as lgb
from catboost import CatBoostClassifier
from queue_features import Panel as OnsetPanel, panel_names, HISTORY, HORIZON
from queue_ongoing import Panel as OngoingPanel, candidate_origins

onset_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
onset_cat = CatBoostClassifier()
onset_cat.load_model(str(ASSETS / "onset.cbm"))
print("onset features:", onset_lgb.num_feature(), "| HORIZON", HORIZON, elapsed())"""))

cells.append(("markdown", """## Policies

`topk` guarantees nesting, so the staircase can never claim a link at step 4 that it drops by
step 6 — a queue grows, it does not un-form inside a window."""))

cells.append(("code", """\
def block(flat):
    return np.asarray(flat).reshape(HORIZON, -1)

def topk_mask(row, k):
    m = np.zeros(row.shape[0], dtype=bool)
    if k <= 0:
        return m
    idx = np.argpartition(row, -min(k, row.shape[0]))[-min(k, row.shape[0]):]
    m[idx] = True
    return m

def policy_masks(prob, threshold, shape=(0.62, 0.79, 1.0), fill_from=4):
    P = block(prob)
    # width at the final step comes from the probability threshold
    row6 = P[-1]
    k = int((row6 >= threshold).sum())
    if k == 0:
        k = 1                                   # every onset window holds a queue
    out = {}

    last = topk_mask(row6, k)
    z = np.zeros_like(P); z[-1] = last
    out["last_only"] = z

    flat = np.zeros_like(P)
    flat[fill_from - 1:] = last[None, :]
    out["flat"] = flat

    stair = np.zeros_like(P)
    for s in range(fill_from - 1, HORIZON):
        share = shape[s - (fill_from - 1)]
        stair[s] = topk_mask(row6, max(1, int(round(k * share))))
    out["staircase"] = stair

    allm = P >= threshold
    if not allm.any():
        allm[-1] = last
    out["all_steps"] = allm
    return out

def block_iou(true_block, pred_block, valid_block):
    v = np.asarray(valid_block).astype(bool)
    t = np.asarray(true_block).astype(bool) & v
    p = np.asarray(pred_block).astype(bool) & v
    u = np.logical_or(t, p).sum()
    return float(np.logical_and(t, p).sum() / u) if u else np.nan

print("ok", elapsed())"""))

cells.append(("markdown", """## Walk February"""))

cells.append(("code", """\
ONSET_PER_DAY = 4
THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55)
SHAPES = {"stair_62_79": (0.62, 0.79, 1.0), "stair_50_75": (0.50, 0.75, 1.0), "stair_80_90": (0.80, 0.90, 1.0)}
rng = np.random.default_rng(20261027)
rows = []

for panel_no, name in enumerate(panel_names()):
    onset_panel, ongoing_panel = OnsetPanel(name, panel_no), OngoingPanel(name, panel_no)
    folder = DATA / "corridors" / name / "train" / "mainline_states"
    files = [p for p in sorted(folder.glob("**/*.parquet")) if p.parent.name.endswith("2031-02")]
    for path in files:
        times, speed, flow, eligible = onset_panel.train_day(path)
        pool = candidate_origins(ongoing_panel, times, speed, eligible).get("queue_onset", [])
        if not pool:
            continue
        pick = rng.choice(np.asarray(pool), size=min(ONSET_PER_DAY, len(pool)), replace=False)
        for t in np.sort(pick):
            origin = pd.Timestamp(times[t])
            X = onset_panel.features(speed[t - HISTORY:t], flow[t - HISTORY:t],
                                     eligible[t - HISTORY:t], origin, "queue_onset")
            prob = (0.5 * onset_lgb.predict(np.asarray(X), num_threads=4)
                    + 0.5 * onset_cat.predict_proba(np.asarray(X), thread_count=4)[:, 1])
            future = speed[t + 1:t + HORIZON + 1]
            cut = ongoing_panel.cut[None, :]
            true = (future <= cut) & np.isfinite(future)
            valid = eligible[t + 1:t + HORIZON + 1] & np.isfinite(future)
            if valid.sum() < 0.6 * valid.size:
                continue
            for thr in THRESHOLDS:
                masks = policy_masks(prob, thr)
                for shape_name, shape in SHAPES.items():
                    masks[shape_name] = policy_masks(prob, thr, shape=shape)["staircase"]
                for policy, mask in masks.items():
                    rows.append((name, policy, thr, block_iou(true, mask, valid)))
    print(f"{name}: done {elapsed()}", flush=True)

detail = pd.DataFrame(rows, columns=["panel", "policy", "threshold", "iou"])
detail.to_csv(WORK / "task2_shape_detail.csv", index=False)
print("windows:", f"{len(detail):,}", elapsed())
"""))

cells.append(("markdown", """## Results (macro mean over panels, as the official aggregation does)"""))

cells.append(("code", """\
per_panel = detail.groupby(["policy", "threshold", "panel"]).iou.mean().reset_index()
table = per_panel.groupby(["policy", "threshold"]).iou.mean().reset_index().sort_values("iou", ascending=False)
print(table.head(18).round(4).to_string(index=False))
print()
print("best per policy:")
print(table.sort_values("iou", ascending=False).groupby("policy").first().sort_values("iou", ascending=False).round(4).to_string())
print()
best = table.iloc[0]
flat = table[(table.policy == "flat") & (table.threshold == 0.40)].iou
print(f"v7 rule (flat @0.40): {float(flat.iloc[0]):.4f}")
print(f"best overall: {best.policy} @{best.threshold} -> {best.iou:.4f}  "
      f"(delta vs v7 {best.iou - float(flat.iloc[0]):+.4f} IoU = {0.30 * 0.5 * (best.iou - float(flat.iloc[0])):+.4f} score)")
"""))


def main():
    out = []
    for kind, src in cells:
        out.append({"cell_type": kind, "metadata": {},
                    "source": [l + "\n" for l in src.splitlines()],
                    **({"outputs": [], "execution_count": None} if kind == "code" else {})})
    nb = {"cells": out,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 4}
    path = HERE / "tf_v6b_shape_audit.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    for c in out:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%%"):
            continue
        compile(src, "cell", "exec")
    print(f"wrote {path.name} with {len(out)} cells; code cells compile")
    meta = {"id": "sanjay0614/tf-v6b-onset-shape",
            "title": "TF V6b Onset Shape", "code_file": path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "competition_sources": ["2026-ieee-big-data-traffic-flow-bench"],
            "dataset_sources": ["anubisyy/trafficflowbench-086517-public-assets"]}
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote kernel-metadata.json")


if __name__ == "__main__":
    main()
