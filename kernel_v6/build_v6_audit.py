"""Offline experiment: how should Task-2 onset masks fill the 6-step horizon?

No leaderboard slot is spent. February 2031 lives in the *train* split, so its labels are
released; a link is queued when its observed speed <= 60% of free-flow speed (fd_parameters).
That gives a true space-time block for every candidate origin, so we can score exactly the
official metric (space-time IoU over the 6 x links block) for competing fill policies.

Why this matters: the shipped inference predicts the onset probability for the last horizon
step only and leaves steps 1-5 at zero (`prediction[-1] = selected`), while the author's
local onset IoU of 0.8925 was measured with probabilities over all six steps. Queue is 30%
of the total score, so a fill-policy error here is expensive.

Policies compared (onset, where history shows no queue yet):
  last_only      what we ship now: only step 6, zero elsewhere
  last_filled    step-6 links persisted back over all six steps
  fill_from_k    step-6 links filling steps k..6 (k = 3,4,5)
  all_steps      threshold the whole block (closest to how the model was validated)
  wedge          step-6 links, expanded upstream (lower link index) for earlier steps
Ongoing windows: current all-step thresholding, swept, as a control.

Output: IoU tables per policy/threshold, macro-averaged over panels, plus per-panel detail.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "scratch" / "kernel_out" / "tf3pub" / "tfb_source"

MODULES = ["config.py", "queue_features.py", "queue_ongoing.py"]

cells = []

cells.append(("markdown", """# Task 2: horizon-fill policy audit (offline, labelled February)

Queue is **30%** of the competition score and is scored as space-time IoU over a
6-step x links block. The shipped pipeline predicts onset probability for the **last step
only** and zeroes the rest. Its published local onset IoU (0.8925) was measured with
probabilities over all six steps, so the shipped rule has never been validated against the
official block metric.

This notebook measures it. February 2031 is in the train split, so labels are released and a
link is queued when speed <= 60% of free-flow speed. Every candidate origin therefore has a
true block, and each fill policy can be scored with the real metric."""))

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
print("data:", DATA)
print("assets:", ASSETS, sorted(p.name for p in ASSETS.glob("*.*") if p.suffix in (".txt", ".cbm"))[:6])
"""))

# module sources, verbatim from the shipped pipeline
for name in MODULES:
    cells.append(("code", f"%%writefile tfb_source/{name}\n" + (SRC / name).read_text(encoding="utf-8")))

cells.append(("markdown", """## Load the shipped models and rebuild windows"""))

cells.append(("code", """\
import lightgbm as lgb
from catboost import CatBoostClassifier
from queue_features import Panel as OnsetPanel, panel_names, HISTORY, HORIZON
from queue_ongoing import Panel as OngoingPanel, candidate_origins

onset_lgb = lgb.Booster(model_file=str(ASSETS / "onset_lgb.txt"))
ongoing_lgb = lgb.Booster(model_file=str(ASSETS / "ongoing_lgb.txt"))
onset_cat = CatBoostClassifier()
onset_cat.load_model(str(ASSETS / "onset.cbm"))
print("features: onset_lgb", onset_lgb.num_feature(), "| ongoing_lgb", ongoing_lgb.num_feature())
print("HISTORY", HISTORY, "HORIZON", HORIZON, elapsed())
"""))

cells.append(("markdown", """## The fill policies

`prob` is the model output for the whole 6 x links block (step-major, so
`prob.reshape(HORIZON, -1)` is (step, link)). Every policy guarantees at least one queued
cell, matching the shipped top-1 fallback (every onset window has a queue in its horizon)."""))

cells.append(("code", """\
def to_block(flat):
    return np.asarray(flat).reshape(HORIZON, -1)

def last_mask(P):
    last = P[-1].copy()
    if not last.any():
        last[int(np.argmax(P[-1]))] = True       # top-1 fallback, as shipped
    return last

def policy_masks(prob, threshold):
    P = to_block(prob) >= threshold
    last = last_mask(P)
    out = {}
    z = np.zeros_like(P); z[-1] = last
    out["last_only"] = z                          # shipped behaviour
    out["last_filled"] = np.repeat(last[None, :], HORIZON, axis=0)
    for k in (3, 4, 5):
        zk = np.zeros_like(P); zk[k - 1:] = last[None, :]
        out[f"fill_from_{k}"] = zk
    zall = P.copy()
    if not zall.any():
        zall[-1, int(np.argmax(P[-1]))] = True
    out["all_steps"] = zall
    # queues grow upstream (towards lower link ids): earlier steps hold a subset
    wedge = np.zeros_like(P)
    span = np.where(last)[0]
    hi = int(span.max()) if len(span) else 0
    for s in range(HORIZON):
        share = (s + 1) / HORIZON
        width = max(1, int(round(len(span) / share))) if len(span) else 1
        idx = [i for i in range(max(0, hi - width + 1), hi + 1) if 0 <= i < P.shape[1]]
        wedge[s, idx] = True
    out["wedge"] = wedge
    return out

def block_iou(true_block, pred_block, valid_block):
    t = true_block & valid_block
    p = pred_block & valid_block
    union = np.logical_or(t, p).sum()
    return float(np.logical_and(t, p).sum() / union) if union else np.nan

print("policies:", list(policy_masks(np.full(HORIZON * 5, 0.9), 0.5)))
"""))

cells.append(("markdown", """## Walk February and score every policy

Windows are drawn with the pipeline's own `candidate_origins`, capped per panel-day so the
sweep stays inside a CPU session. Truth and validity come from the released observations."""))

cells.append(("code", """\
ONSET_PER_DAY, ONGOING_PER_DAY = 4, 2
ONSET_THRESHOLDS = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
ONGOING_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)

rng = np.random.default_rng(20261027)
rows = []      # (panel, condition, policy, threshold, iou)
stats = {"onset_step_pos": np.zeros(HORIZON), "onset_n": 0,
         "ongoing_step_pos": np.zeros(HORIZON), "ongoing_n": 0}

for panel_no, name in enumerate(panel_names()):
    onset_panel, ongoing_panel = OnsetPanel(name, panel_no), OngoingPanel(name, panel_no)
    folder = DATA / "corridors" / name / "train" / "mainline_states"
    files = [p for p in sorted(folder.glob("**/*.parquet")) if p.parent.name.endswith("2031-02")]
    links = ongoing_panel.links
    for path in files:
        times, speed, flow, eligible = onset_panel.train_day(path)
        cand = candidate_origins(ongoing_panel, times, speed, eligible)
        for condition, cap, thresholds in (
            ("queue_onset", ONSET_PER_DAY, ONSET_THRESHOLDS),
            ("queue_ongoing", ONGOING_PER_DAY, ONGOING_THRESHOLDS),
        ):
            pool = cand.get(condition, [])
            if not pool:
                continue
            pick = rng.choice(np.asarray(pool), size=min(cap, len(pool)), replace=False)
            for t in np.sort(pick):
                origin = pd.Timestamp(times[t])
                if condition == "queue_onset":
                    X = onset_panel.features(speed[t - HISTORY:t], flow[t - HISTORY:t],
                                             eligible[t - HISTORY:t], origin, condition)
                    prob = (0.5 * onset_lgb.predict(np.asarray(X), num_threads=4)
                            + 0.5 * onset_cat.predict_proba(np.asarray(X), thread_count=4)[:, 1])
                else:
                    obs = np.nan_to_num(eligible[t - HISTORY:t], nan=0).astype(bool)
                    X = ongoing_panel.features(speed[t - HISTORY:t], flow[t - HISTORY:t], obs, origin, condition)
                    prob = ongoing_lgb.predict(np.asarray(X), num_threads=4)
                future = speed[t + 1:t + HORIZON + 1]
                cut = ongoing_panel.cut[None, :]
                true = (future <= cut) & np.isfinite(future)
                valid = (eligible[t + 1:t + HORIZON + 1] & np.isfinite(future))
                if valid.sum() < 0.6 * valid.size:
                    continue
                key = "onset_step_pos" if condition == "queue_onset" else "ongoing_step_pos"
                stats[key] += (true & valid).sum(axis=1)
                stats[key.replace("step_pos", "n")] += 1
                for thr in thresholds:
                    masks = policy_masks(prob, thr)
                    if condition == "queue_ongoing":
                        masks = {"all_steps": masks["all_steps"], "last_only": masks["last_only"]}
                    for policy, mask in masks.items():
                        rows.append((name, condition, policy, thr,
                                     block_iou(true, mask, valid)))
    print(f"{name}: windows scored, {elapsed()}", flush=True)

detail = pd.DataFrame(rows, columns=["panel", "condition", "policy", "threshold", "iou"])
detail.to_csv(WORK / "task2_fill_detail.csv", index=False)
print("windows:", f"{len(detail):,}", elapsed())
print("mean positives per step (onset)  ", (stats["onset_step_pos"] / max(stats["onset_n"], 1)).round(2))
print("mean positives per step (ongoing)", (stats["ongoing_step_pos"] / max(stats["ongoing_n"], 1)).round(2))
print("windows: onset", stats["onset_n"], "ongoing", stats["ongoing_n"])
"""))

cells.append(("markdown", """## Results

Macro-averaged over panels (the official aggregation is an equal mean over panels and the two
conditions, so panel symmetry matters more than window count)."""))

cells.append(("code", """\
def summarise(condition):
    d = detail[detail.condition == condition]
    per_panel = d.groupby(["policy", "threshold", "panel"]).iou.mean().reset_index()
    table = (per_panel.groupby(["policy", "threshold"]).iou.mean()
             .reset_index().sort_values("iou", ascending=False))
    return table

for condition in ("queue_onset", "queue_ongoing"):
    print(f"===== {condition}: top 12 of policy x threshold =====")
    print(summarise(condition).head(12).to_string(index=False))
    print()

print("===== onset: best threshold per policy =====")
o = summarise("queue_onset")
print(o.groupby("policy").first().sort_values("iou", ascending=False).round(4).to_string())

shipped = o[(o.policy == "last_only") & (o.threshold == 0.55)].iou
best = o.iou.max()
print(f"\\nshipped onset rule (last_only @0.55): {float(shipped.iloc[0]):.4f}")
print(f"best onset rule: {best:.4f}  ->  delta {best - float(shipped.iloc[0]):+.4f} IoU")
print(f"implied overall delta if it holds: {0.30 * 0.5 * (best - float(shipped.iloc[0])):+.4f} score")
"""))

cells.append(("markdown", """## Notes

Absolute IoU here is a February proxy: the organizer's hidden windows are chosen by their own
rule, and their truth is organizer-held. What transfers is the *ordering* of fill policies,
because every policy is scored on identical windows, truth and model probabilities."""))


def main():
    out = []
    for kind, src in cells:
        out.append({
            "cell_type": kind,
            "metadata": {},
            "source": [l + "\n" for l in src.splitlines()],
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        })
    nb = {"cells": out,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 4}
    path = HERE / "tf_v6_task2_fill_audit.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    for c in out:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith("%%"):
            continue
        compile(src, "cell", "exec")
    print(f"wrote {path.name} with {len(out)} cells; plain code cells compile")

    meta = {
        "id": "sanjay0614/tf-v6-task2-fill-audit",
        "title": "TF V6 Task2 Fill Audit",
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["2026-ieee-big-data-traffic-flow-bench"],
        "dataset_sources": ["anubisyy/trafficflowbench-086517-public-assets"],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote kernel-metadata.json")


if __name__ == "__main__":
    main()
