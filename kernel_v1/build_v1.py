"""Builds tf_v1_task1_baseline.ipynb.

Run:  python build_v1.py
Then: kaggle kernels push   (from this folder, with kernel-metadata.json present)

The notebook reproduces the host's Task 1 historical-mean baseline
(src/task1/baseline_task1_historical_mean.py) and scores it on the train split
the way src/task1/score_task1.py does, on all ten panels.
"""
import json
from pathlib import Path

nb_version = "v1"

CELLS = []

CELLS.append(("markdown", """# TF V1 — Task 1 baseline, reproduced and self-scored on `train`

Reproduces the host's published baseline (weekday × time-of-day historical mean per link),
then scores it on `train` with the host's own evaluator logic, on all ten panels.

Deliberately **not** a submission: this run exists to build and trust the scoring harness.

Gate: our per-panel `S_state` numbers must be plausible against the host's commentary, and the
harness must produce a full panel/family/overall table for every later version to be judged by.

Facts encoded here, taken from the host repository (github.com/jacky850/trafficflowbench-public):

- profile: one mean speed and flow per `(link_id, weekday, 5-min-slot-of-day)`, learned on the
  unmasked `train` observation layer, cells only where `pct_observed >= 75` and both channels
  are present; link-wide fallback where a slot has no observations;
- targets: masked-layer cells whose measured channels are null, joined (1:1) against the
  unmasked truth restricted to `is_score_eligible`;
- `S_speed = max(0, 1 - RMSE_speed/25)`, `S_flow = max(0, 1 - (RMSE_flow/lane)/600)`,
  `S_state = 0.54*S_speed + 0.46*S_flow`;
- aggregation: regime → panel → family → overall, all macro (unweighted) means;
- missing predictions count as zero and stay in the denominator.
"""))

CELLS.append(("code", """\
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = next(p for p in (Path("/kaggle/input") / "2026-ieee-big-data-traffic-flow-bench",
                             Path("/kaggle/input") / "competitions" / "2026-ieee-big-data-traffic-flow-bench")
                  if p.exists())
RELEASE = DATA_ROOT / "kaggle_public"
assert RELEASE.exists(), f"release root not found: {RELEASE}"

PANEL_FAMILY = {
    "D7_I10_E": "D7_I10",   "D7_I10_W": "D7_I10",
    "D7_I210_E": "D7_I210", "D7_I210_W": "D7_I210",
    "D7_I405_N": "D7_I405", "D7_I405_S": "D7_I405",
    "D12_I5_N": "D12_I5",   "D12_I5_S": "D12_I5",
    "D12_I405_N": "D12_I405", "D12_I405_S": "D12_I405",
}
PANELS = list(PANEL_FAMILY)
REGIMES = ("R1", "R2", "R3")
SPEED_NORMALIZER = 25.0
FLOW_NORMALIZER = 600.0
SPEED_WEIGHT = 0.54
FLOW_WEIGHT = 0.46

print("release root:", RELEASE)
print("panels:", len(PANELS))
"""))

CELLS.append(("markdown", """## Lane counts

Flow is scored **per lane**: both submitted and true flow are divided by the station's lane count
before the RMSE, which is what makes `FLOW_NORMALIZER = 600` meaningful. Lane counts come from
`fd_parameters.csv` (preferred: it carries `station_id`) with `links.csv` as the fallback; zone
connectors marked `lanes = 99` are excluded. Ported from the host's `station_lanes()` /
`lane_vector()`."""))

CELLS.append(("code", """\
def station_lanes(panel_dir: Path) -> dict:
    lanes_by_key = {}

    def absorb(frame: pd.DataFrame, key: str) -> None:
        if key not in frame.columns or "lanes" not in frame.columns:
            return
        lanes = pd.to_numeric(frame["lanes"], errors="coerce")
        valid = lanes.notna() & (lanes >= 1) & (lanes <= 16)
        lanes_by_key.update(zip(frame[key][valid].astype(str), lanes[valid].astype(float)))

    fd = panel_dir / "network" / "fd_parameters.csv"
    links = panel_dir / "network" / "links.csv"
    if links.exists():
        absorb(pd.read_csv(links, dtype={"link_id": str}), "link_id")
    if fd.exists():
        frame = pd.read_csv(fd, dtype={"station_id": str, "link_id": str})
        absorb(frame, "link_id")
        absorb(frame, "station_id")
    if not lanes_by_key:
        raise FileNotFoundError(f"no lane count under {panel_dir / 'network'}")
    return lanes_by_key


def lane_vector(stations: pd.Series, links: pd.Series, lanes_by_key: dict) -> np.ndarray:
    by_station = stations.astype(str).map(lanes_by_key)
    by_link = links.astype(str).map(lanes_by_key)
    return pd.to_numeric(by_station.fillna(by_link), errors="coerce").fillna(1.0).clip(lower=1.0).to_numpy(float)
"""))

CELLS.append(("markdown", """## The profile

One weekday × 288-slot mean per link, learned on the unmasked `train` layer. Profile cells require
`pct_observed >= 75` and finite speed and flow. Slot with no observations falls back to the
link-wide mean."""))

CELLS.append(("code", """\
def files(panel_dir: Path, split: str):
    direct = sorted((panel_dir / split / "mainline_states").glob("**/*.parquet"))
    if direct:
        return direct
    return sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))


def masked_files(panel_dir: Path, split: str, regime: str):
    return sorted(
        (panel_dir / split / "mainline_states_masked" / f"mask_regime={regime}").glob("*.parquet")
    )


def slot_values(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    weekday = ts.dt.weekday.to_numpy(np.int16)
    tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int16)
    return weekday, tod


def build_profile(panel: str, panel_dir: Path):
    train_files = files(panel_dir, "train")
    if not train_files:
        raise RuntimeError(f"{panel}: no train observation partitions")
    first = pd.read_parquet(train_files[0], columns=["link_id"])
    link_ids = sorted(first.link_id.astype(str).unique())
    link_index = {lk: i for i, lk in enumerate(link_ids)}
    n_slots = 7 * 288
    speed_sum = np.zeros((len(link_ids), n_slots)); flow_sum = np.zeros_like(speed_sum)
    speed_cnt = np.zeros((len(link_ids), n_slots), np.int64); flow_cnt = np.zeros_like(speed_cnt)
    t0 = time.time()
    for i, path in enumerate(train_files, 1):
        frame = pd.read_parquet(
            path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"]
        )
        li = frame.link_id.astype(str).map(link_index).to_numpy(np.int64)
        wd, tod = slot_values(frame)
        slot = wd * 288 + tod
        valid = (
            pd.to_numeric(frame.pct_observed, errors="coerce").ge(75)
            & frame.speed_kmh.notna() & frame.flow_vph.notna()
        ).to_numpy()
        sp = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(float)
        fl = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(float)
        ok_sp = valid & np.isfinite(sp)
        ok_fl = valid & np.isfinite(fl)
        np.add.at(speed_sum, (li[ok_sp], slot[ok_sp]), sp[ok_sp])
        np.add.at(speed_cnt, (li[ok_sp], slot[ok_sp]), 1)
        np.add.at(flow_sum, (li[ok_fl], slot[ok_fl]), fl[ok_fl])
        np.add.at(flow_cnt, (li[ok_fl], slot[ok_fl]), 1)
        if i % 50 == 0:
            print(f"[{panel}] profile {i}/{len(train_files)} ({time.time()-t0:.0f}s)", flush=True)
    speed_mean = np.divide(speed_sum, np.maximum(speed_cnt, 1),
                           out=np.zeros_like(speed_sum), where=speed_cnt > 0)
    flow_mean = np.divide(flow_sum, np.maximum(flow_cnt, 1),
                          out=np.zeros_like(flow_sum), where=flow_cnt > 0)
    speed_fb = np.divide(speed_sum.sum(1), np.maximum(speed_cnt.sum(1), 1))
    flow_fb = np.divide(flow_sum.sum(1), np.maximum(flow_cnt.sum(1), 1))
    return {
        "link_index": link_index,
        "speed_mean": speed_mean, "flow_mean": flow_mean,
        "speed_fb": speed_fb, "flow_fb": flow_fb,
        "speed_cnt": speed_cnt, "flow_cnt": flow_cnt,
        "profile_seconds": time.time() - t0,
    }
"""))

CELLS.append(("markdown", """## Score on `train`, following `score_task1.py`

For each regime directory: targets are the cells whose `speed_kmh` **and** `flow_vph` are null in
the masked layer; they are joined 1:1 against the unmasked truth restricted to
`is_score_eligible == True`; predictions come from the profile inline. Missing predictions would
enter as zero — we never have any, because every target is predicted here."""))

CELLS.append(("code", """\
def evaluate_panel(panel: str, release: Path) -> list:
    panel_dir = release / "corridors" / panel
    prof = build_profile(panel, panel_dir)
    lanes_by_key = station_lanes(panel_dir)
    link_index = prof["link_index"]
    truth_by_name = {p.name: p for p in files(panel_dir, "train")}
    rows = []
    for regime in REGIMES:
        sq_speed = sq_flow = 0.0
        n_target = n_missed = 0
        t0 = time.time()
        for masked_path in masked_files(panel_dir, "train", regime):
            truth_path = truth_by_name.get(masked_path.name)
            if truth_path is None:
                continue
            frame = pd.read_parquet(
                truth_path,
                columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"],
            )
            blanked = pd.read_parquet(
                masked_path, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"]
            )
            blanked = blanked[blanked.speed_kmh.isna() & blanked.flow_vph.isna()][
                ["timestamp", "station_id", "link_id"]
            ]
            if blanked.empty:
                continue
            eligible = frame[frame.is_score_eligible.astype(bool)]
            targets = eligible.merge(blanked, on=["timestamp", "station_id", "link_id"], validate="one_to_one")
            if targets.empty:
                continue
            li = targets.link_id.astype(str).map(link_index).to_numpy(np.int64)
            wd, tod = slot_values(targets)
            slot = wd * 288 + tod
            has_slot = prof["speed_cnt"][li, slot] > 0
            pred_speed = np.where(has_slot, prof["speed_mean"][li, slot],
                                  prof["speed_fb"][li])
            has_slot_f = prof["flow_cnt"][li, slot] > 0
            pred_flow = np.where(has_slot_f, prof["flow_mean"][li, slot],
                                 prof["flow_fb"][li])
            true_speed = pd.to_numeric(targets.speed_kmh, errors="coerce").to_numpy(float)
            true_flow = pd.to_numeric(targets.flow_vph, errors="coerce").to_numpy(float)
            unusable = ~np.isfinite(true_speed) | ~np.isfinite(true_flow)
            n_missed += int(unusable.sum())
            usable = ~unusable
            ds = pred_speed[usable] - true_speed[usable]
            lane = lane_vector(targets.station_id, targets.link_id, lanes_by_key)
            dq = (pred_flow[usable] - true_flow[usable]) / lane[usable]
            sq_speed += float(np.sum(ds * ds))
            sq_flow += float(np.sum(dq * dq))
            n_target += int(usable.sum())
        if n_target == 0:
            rmse_s = rmse_f = float("nan")
            s_speed = s_flow = s_state = 0.0
        else:
            rmse_s = float(np.sqrt(sq_speed / n_target))
            rmse_f = float(np.sqrt(sq_flow / n_target))
            s_speed = max(0.0, 1.0 - rmse_s / SPEED_NORMALIZER)
            s_flow = max(0.0, 1.0 - rmse_f / FLOW_NORMALIZER)
            s_state = SPEED_WEIGHT * s_speed + FLOW_WEIGHT * s_flow
        print(f"[{panel}] {regime}: n={n_target} RMSE_s={rmse_s:.2f} RMSE_f/lane={rmse_f:.1f} "
              f"S_state={s_state:.4f} ({time.time()-t0:.0f}s)", flush=True)
        rows.append({"panel": panel, "family_id": PANEL_FAMILY[panel], "regime": regime,
                     "n_target_cells": n_target, "n_invalid_truth": n_missed,
                     "rmse_speed_kmh": rmse_s, "rmse_flow_per_lane": rmse_f,
                     "S_speed": s_speed, "S_flow": s_flow, "S_state": s_state})
    panel_mean = float(np.mean([r["S_state"] for r in rows]))
    rows.append({"panel": panel, "family_id": PANEL_FAMILY[panel], "regime": "macro_mean",
                 "n_target_cells": sum(r["n_target_cells"] for r in rows),
                 "n_invalid_truth": sum(r["n_invalid_truth"] for r in rows),
                 "rmse_speed_kmh": None, "rmse_flow_per_lane": None,
                 "S_speed": None, "S_flow": None, "S_state": panel_mean})
    print(f"[{panel}] panel S_state macro mean = {panel_mean:.4f}; "
          f"profile took {prof['profile_seconds']:.0f}s to build", flush=True)
    return rows
"""))

CELLS.append(("markdown", """## Run all ten panels

Timing note: `train` is 273 days per panel (× two layers read per day) and ten panels. This is the
one measurement of what a full self-scored pass costs, which decides how v2 experiments get
budgeted."""))

CELLS.append(("code", """\
all_rows, total_t0 = [], time.time()
for panel in PANELS:
    t0 = time.time()
    all_rows.extend(evaluate_panel(panel, RELEASE))
    print(f"=== {panel} done in {time.time()-t0:.0f}s ===", flush=True)

detail = pd.DataFrame(all_rows)
detail.to_csv("/kaggle/working/v1_task1_s_state_detail.csv", index=False)

panel_rows = detail[detail.regime == "macro_mean"].copy()
families = {}
for fam, grp in panel_rows.groupby("family_id"):
    families[fam] = float(grp.S_state.mean())
overall = float(np.mean(list(families.values())))

summary = {
    "panel_S_state": {r.panel: r.S_state for r in panel_rows.itertuples()},
    "family_S_state": families,
    "overall_S_state": overall,
    "wall_clock_seconds": time.time() - total_t0,
}
with open("/kaggle/working/v1_task1_scores.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print()
print("family S_state (macro mean over regimes, then panels):")
for fam, v in sorted(families.items()):
    print(f"  {fam:10s} {v:.4f}")
print()
print(f"OVERALL S_state = {overall:.4f}")
print(f"wall clock: {summary['wall_clock_seconds']/60:.1f} min")
summary
"""))

CELLS.append(("markdown", """## What to read off the result

- **Family spread.** The official score is a macro mean over the five families, so a weak family
  drags the total as hard as a strong one. The weakest family is where v2 work starts.
- **Regime drop.** Going R1 → R3 is the same model with more holes; if `S_state` collapses between
  regimes, the profile is brittle rather than the answer being hard.
- **Wall clock.** The total here sets the experiment budget for v2: a full pass is one unit.

Gate to pass before v1 is accepted as done: numbers are non-degenerate (a constant-profile model
should land well below a plausible reconstruction, typically `S_state` in the 0.3–0.7 band), all
ten panels scored, and no panel pattern suggests our join is broken (e.g. `n_target_cells` equal
across regimes would mean the regime directories are being misread)."""))


def main():
    cells = []
    for kind, src in CELLS:
        cells.append({
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        })
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = Path(__file__).resolve().parent / "tf_v1_task1_baseline.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} with {len(cells)} cells")


if __name__ == "__main__":
    main()
