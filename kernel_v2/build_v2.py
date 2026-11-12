"""Builds tf_v2_first_submission.ipynb.

Run:  python build_v2.py
Then: kaggle kernels push -p .

Produces the first complete submission file:
  state:  historical-mean profile (v1's method) for validation + private
  queue:  persistence baseline, mirroring the host's build_task2_persistence_submission.py
  odme:   per-panel NNLS on the link counts with the weak prior as regularisation
  merged: streamed join onto submission_key.csv in template order, per merge_submissions.py
"""
import json
from pathlib import Path

CELLS = []

CELLS.append(("markdown", """# TF V2 — first complete submission

One run, three tasks, one merged upload file. This is the pipeline check the plan originally
scheduled as V5, pulled forward: with five submissions a day available, the cheapest lesson — a
complete file that scores on every task — should be bought immediately.

- **state** (0.35 + 0.15 via physics): historical-mean profile learned on `train`, predicted for
  every Task 1 target cell in `validation` and `private`, using the per-corridor templates.
- **queue** (0.30): the host's persistence baseline, mirrored — repeat the queue state at the
  last history step (T-5) through all six steps; threshold at released `v_cut` or 0.6 × free speed.
- **odme** (0.20): per panel and split, non-negative least squares on `A x = c` with the weak
  prior as a ridge regulariser. Tiny systems: one aggregated departure time per panel/split.

Merged with the host's `merge_submissions.py` logic, streamed in 1M-row chunks over
`submission_key.csv` (643 MB). Assertions: exactly 6,980,503 rows, no nulls, no unmatched keys."""))

CELLS.append(("code", """\
import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = next(p for p in (Path("/kaggle/input") / "2026-ieee-big-data-traffic-flow-bench" / "kaggle_public",
                           Path("/kaggle/input") / "competitions" / "2026-ieee-big-data-traffic-flow-bench" / "kaggle_public")
               if p.exists())
WORK = Path("/kaggle/working")
assert RELEASE.exists(), RELEASE

PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
          "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S",
          "D12_I405_N", "D12_I405_S"]
SPLITS = ["validation", "private"]
N_STATES_EXPECTED = {"validation": 31, "private": 30}  # days per split, for margins only
print("release root:", RELEASE)
"""))

CELLS.append(("markdown", """## Profiles and lane map

Identical to v1: weekday × 288-slot means per link, learned on the unmasked `train` observation
layer with the `pct_observed >= 75` and finite-value rule, link-wide fallback for empty slots."""))

CELLS.append(("code", """\
def slot_values(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    return ts.dt.weekday.to_numpy(np.int16), (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int16)


def build_profile(panel_dir: Path):
    train_files = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
    assert train_files, f"no train files under {panel_dir}"
    first = pd.read_parquet(train_files[0], columns=["link_id"])
    link_ids = sorted(first.link_id.astype(str).unique())
    link_index = {lk: i for i, lk in enumerate(link_ids)}
    n_slots = 7 * 288
    speed_sum = np.zeros((len(link_ids), n_slots)); flow_sum = np.zeros_like(speed_sum)
    speed_cnt = np.zeros((len(link_ids), n_slots), np.int64); flow_cnt = np.zeros_like(speed_cnt)
    t0 = time.time()
    for path in train_files:
        frame = pd.read_parquet(path, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed"])
        li = frame.link_id.astype(str).map(link_index).to_numpy(np.int64)
        wd, tod = slot_values(frame)
        slot = wd * 288 + tod
        valid = (pd.to_numeric(frame.pct_observed, errors="coerce").ge(75)
                 & frame.speed_kmh.notna() & frame.flow_vph.notna()).to_numpy()
        sp = pd.to_numeric(frame.speed_kmh, errors="coerce").to_numpy(float)
        fl = pd.to_numeric(frame.flow_vph, errors="coerce").to_numpy(float)
        ok_sp, ok_fl = valid & np.isfinite(sp), valid & np.isfinite(fl)
        np.add.at(speed_sum, (li[ok_sp], slot[ok_sp]), sp[ok_sp])
        np.add.at(speed_cnt, (li[ok_sp], slot[ok_sp]), 1)
        np.add.at(flow_sum, (li[ok_fl], slot[ok_fl]), fl[ok_fl])
        np.add.at(flow_cnt, (li[ok_fl], slot[ok_fl]), 1)
    speed_mean = np.divide(speed_sum, np.maximum(speed_cnt, 1), out=np.zeros_like(speed_sum), where=speed_cnt > 0)
    flow_mean = np.divide(flow_sum, np.maximum(flow_cnt, 1), out=np.zeros_like(flow_sum), where=flow_cnt > 0)
    speed_fb = np.divide(speed_sum.sum(1), np.maximum(speed_cnt.sum(1), 1))
    flow_fb = np.divide(flow_sum.sum(1), np.maximum(flow_cnt.sum(1), 1))
    print(f"    profile built in {time.time()-t0:.0f}s", flush=True)
    return {"link_index": link_index, "speed_mean": speed_mean, "flow_mean": flow_mean,
            "speed_fb": speed_fb, "flow_fb": flow_fb, "speed_cnt": speed_cnt, "flow_cnt": flow_cnt}
"""))

CELLS.append(("markdown", """## Task 1 — state predictions written per panel/split

Predicted inline and appended to one combined CSV. Speeds and flows are clipped at zero; the
profile can produce tiny negatives for flows near zero, which is not a legal speed."""))

CELLS.append(("code", """\
STATE_HEADER = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
state_path = WORK / "state_submission.csv"
first_write = True
t_all = time.time()
profiles = {}
for panel in PANELS:
    panel_dir = RELEASE / "corridors" / panel
    profiles[panel] = build_profile(panel_dir)
    prof = profiles[panel]
    li_map = prof["link_index"]
    for split in SPLITS:
        tpl = pd.read_csv(RELEASE / "task1" / panel / split / "sample_submission_state.csv")
        tpl["link_id"] = tpl.link_id.astype(str)
        li = tpl.link_id.map(li_map)
        unknown = int(li.isna().sum())
        assert unknown == 0, f"{panel}/{split}: {unknown} template links not in train profile"
        li = li.to_numpy(np.int64)
        wd, tod = slot_values(tpl)
        slot = wd * 288 + tod
        has_s, has_f = prof["speed_cnt"][li, slot] > 0, prof["flow_cnt"][li, slot] > 0
        tpl["speed_kmh"] = np.clip(np.where(has_s, prof["speed_mean"][li, slot], prof["speed_fb"][li]), 0, None).round(3)
        tpl["flow_vph"] = np.clip(np.where(has_f, prof["flow_mean"][li, slot], prof["flow_fb"][li]), 0, None).round(3)
        out = tpl[STATE_HEADER]
        out.to_csv(state_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        print(f"[{panel}/{split}] state rows {len(out):,}", flush=True)
print(f"state stage total {time.time()-t_all:.0f}s")

# free it: the big grids are not needed by later stages
profiles.clear()
print("state CSV bytes:", state_path.stat().st_size)
"""))

CELLS.append(("markdown", """## Task 2 — persistence, mirrored from the host

For every window: take the last history step (`T-5`), threshold its eligible speeds at the
released `v_cut` (else 0.6 × the link free speed, 105 × 0.6 fallback), and repeat that link-level
queue state across all six forecast steps."""))

CELLS.append(("code", """\
def queue_thresholds(panel_dir: Path, links: pd.DataFrame) -> dict:
    fallback = pd.to_numeric(links["free_speed_kmh"], errors="coerce").fillna(105.0) * 0.60
    values = dict(zip(links["link_id"].astype(str), fallback.astype(float)))
    fd = panel_dir / "network" / "fd_parameters.csv"
    if fd.exists():
        frame = pd.read_csv(fd)
        if "v_cut" in frame.columns:
            for row in frame.dropna(subset=["link_id", "v_cut"]).itertuples(index=False):
                values[str(row.link_id)] = float(row.v_cut)
    return values


t_all = time.time()
queue_path = WORK / "queue_submission.csv"
HIST_COLUMNS = ["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]
q_first = True
for panel in [p for p in PANELS if (RELEASE / "task2" / p / "validation" / "window_index.csv").exists()]:
    panel_dir = RELEASE / "corridors" / panel
    links = pd.read_csv(panel_dir / "network" / "links.csv")
    links["link_id"] = links.link_id.astype(str)
    thresholds = queue_thresholds(panel_dir, links)
    for split in SPLITS:
        windows = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
        history = pd.read_parquet(RELEASE / "task2" / panel / split / "window_history.parquet", columns=HIST_COLUMNS)
        template = pd.read_csv(RELEASE / "task2" / panel / split / "sample_submission_queue.csv",
                               usecols=["window_id", "timestamp", "link_id"], dtype={"window_id": str, "link_id": str})
        history["window_id"] = history.window_id.astype(str)
        history["link_id"] = history.link_id.astype(str)
        history["speed_kmh"] = pd.to_numeric(history.speed_kmh, errors="coerce")
        last_time = history.groupby("window_id").timestamp.transform("max")
        at = history[history.timestamp.eq(last_time)].copy()
        at["queue_now"] = at.speed_kmh <= at.link_id.map(thresholds).fillna(63.0)
        at["queue_now"] &= at.is_score_eligible.astype(bool)
        q = at.groupby("link_id").queue_now.any()
        template["queue_pred"] = template.link_id.map(q).fillna(False).astype(int)
        template.to_csv(queue_path, mode="w" if q_first else "a", header=q_first, index=False)
        q_first = False
        print(f"[{panel}/{split}] queue rows {len(template):,} "
              f"(positive {int(template.queue_pred.sum()):,})", flush=True)
print(f"queue stage total {time.time()-t_all:.0f}s; bytes {queue_path.stat().st_size}")
"""))

CELLS.append(("markdown", """## Task 4 — NNLS with prior regularisation, per panel and split

One aggregated departure time per panel/split; `A` is the path-link incidence restricted to the
counted links, `c` the released counts, and the prior enters as ridge rows `lambda * x ~ lambda * x0`,
solved by `scipy.optimize.nnls`. Paths that carry no counted link keep their prior value."""))

CELLS.append(("code", """\
from scipy.optimize import nnls

t_all = time.time()
odme_path = WORK / "odme_submission.csv"
o_first = True
for panel in PANELS:
    for split in SPLITS:
        d = RELEASE / "task4" / panel / split
        counts = pd.read_csv(d / "synthetic_link_counts.csv")
        prior = pd.read_csv(d / "synthetic_weak_prior.csv")
        template = pd.read_csv(d / "sample_submission_path_flow.csv")
        counts["link_id"] = counts.link_id.astype(str)
        prior["path_id"] = prior.path_id.astype(str)
        template["path_id"] = template.path_id.astype(str)
        inc = pd.read_csv(RELEASE / "corridors" / panel / "network" / "path_link_incidence.csv")
        # presence/absence style: rows are (link_id, path_id) pairs, no value column
        inc["use"] = 1.0
        A = inc.pivot_table(index="path_id", columns="link_id", values="use", aggfunc="max", fill_value=0.0)
        A = A.reindex(sorted(A.columns), axis=1).fillna(0.0)
        counted = [lk for lk in A.columns if lk in set(counts.link_id)]
        x = prior.set_index("path_id").path_flow.reindex(A.index).to_numpy(float)
        x = np.nan_to_num(x)
        if counted:
            sub = A[counted]
            c = (counts.set_index("link_id")["count"]
                 .reindex(counted).fillna(0.0).to_numpy(float))
            lam = 0.10 * float(np.mean(np.abs(c)))
            # unknowns are the PATH flows: transpose (paths x links) -> (links x paths),
            # restrict to counted links, add ridge rows pulling f towards the prior x
            A_sys = np.vstack([sub.to_numpy(float).T, lam * np.eye(len(A.index))])
            c_sys = np.concatenate([c, lam * x])
            sol, _ = nnls(A_sys, c_sys)
        else:
            sol = x
        answer = pd.DataFrame({"path_id": A.index, "path_flow": np.clip(sol, 0, None)})
        merged = template[["panel", "departure_time", "path_id"]].merge(answer, on="path_id", how="left")
        missing = int(merged.path_flow.isna().sum())
        merged["path_flow"] = merged.path_flow.fillna(0.0).round(3)
        merged.to_csv(odme_path, mode="w" if o_first else "a", header=o_first, index=False)
        o_first = False
        res_expr = (f"| nnls residual {float(np.sqrt(np.mean((sub.to_numpy(float).T @ sol - c) ** 2))):.1f}"
                    if counted else "")
        print(f"[{panel}/{split}] paths {len(merged):,} | counted links {len(counted)} "
              f"{res_expr} | gap rows -> prior fallback {missing}", flush=True)
print(f"odme stage total {time.time()-t_all:.0f}s; bytes {odme_path.stat().st_size}")
"""))

CELLS.append(("markdown", """## Merge onto `submission_key.csv`, streamed

The host's `merge_submissions.py` logic, verbatim in behaviour: stream the key in 1M-row chunks,
restrict each chunk's rows by task, build the MultiIndex the task's key expects, reindex onto it,
and write the upload file in template order. Unmatched keys become 0 with a warning — we assert
there are none, because our per-task files were built from the same templates."""))

CELLS.append(("code", """\
CHUNK = 1_000_000
KEYS = {"state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
        "queue": ["window_id", "timestamp", "link_id"],
        "odme": ["panel", "departure_time", "path_id"]}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
OUT_COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]

t_all = time.time()
tables = {"state": pd.read_csv(WORK / "state_submission.csv", dtype={"panel": str, "station_id": str, "link_id": str, "mask_regime": str}),
          "queue": pd.read_csv(WORK / "queue_submission.csv", dtype={"window_id": str, "link_id": str}),
          "odme": pd.read_csv(WORK / "odme_submission.csv", dtype={"panel": str, "path_id": str})}
for task, frame in tables.items():
    for col in KEYS[task]:
        if col == "timestamp":
            frame[col] = pd.to_datetime(frame[col], utc=True)
        else:
            frame[col] = frame[col].astype(str)
    for col in VALUES[task]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    dupes = int(frame.duplicated(KEYS[task]).sum())
    assert dupes == 0, f"{task}: {dupes} duplicate keys"
    tables[task] = frame.set_index(KEYS[task])
    print(f"{task}: {len(frame):,} rows loaded for merge", flush=True)

key_path = RELEASE / "submission_key.csv"
gaps = {t: 0 for t in tables}
written = 0
first = True
out_path = WORK / "submission.csv"
for chunk in pd.read_csv(key_path, chunksize=CHUNK, dtype=str):
    chunk["timestamp"] = pd.to_datetime(chunk.timestamp, utc=True, errors="coerce")
    out = pd.DataFrame({"submission_id": chunk.submission_id.astype("int64"),
                        "task": chunk.task.astype(str)})
    for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[col] = 0.0
    for task, table in tables.items():
        rows = out.task == task
        if not rows.any():
            continue
        wanted = pd.MultiIndex.from_frame(chunk.loc[rows, KEYS[task]])
        found = table.reindex(wanted)
        for col in VALUES[task]:
            values = found[col].to_numpy()
            gaps[task] += int(pd.isna(values).sum())
            out.loc[rows, col] = pd.Series(values).fillna(0.0).to_numpy()
    out[OUT_COLUMNS].to_csv(out_path, mode="w" if first else "a", header=first, index=False)
    first = False
    written += len(out)
    print(f"  {written:,} rows merged ({time.time()-t_all:.0f}s)", end="\\r", flush=True)

print()
print(f"merged {written:,} rows in {time.time()-t_all:.0f}s")
for task, n in gaps.items():
    print(f"gap rows for {task}: {n:,}")
assert written == 6_980_503, f"merged row count {written:,} != 6,980,503"
assert all(v == 0 for v in gaps.values()), f"unmatched keys: {gaps}"
print("MERGE OK")
"""))

CELLS.append(("markdown", """## Final validation of the upload file

No blank cells anywhere, correct row count, correct header, and per-task value statistics that are
not degenerate (state speeds in a physical range, queue only 0/1, path flows non-negative with a
sane spread)."""))

CELLS.append(("code", """\
t0 = time.time()
head = pd.read_csv(out_path, nrows=5)
print("head:")
print(head.to_string(index=False))
tail = pd.read_csv(out_path, skiprows=6_980_498 - 2, header=None)
print("tail:")
print(tail.to_string(index=False))

total_rows, nulls, tasks_counts = 0, 0, {}
speeds, flows, queues, pathflow = [], [], [], []
for chunk in pd.read_csv(out_path, chunksize=2_000_000, usecols=["task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]):
    total_rows += len(chunk)
    nulls += int(chunk.isna().sum().sum())
    for t, group in chunk.groupby("task"):
        tasks_counts[t] = tasks_counts.get(t, 0) + len(group)
        if t == "state":
            speeds.extend([group.speed_kmh.min(), group.speed_kmh.max()])
            flows.extend([group.flow_vph.min(), group.flow_vph.max()])
        elif t == "queue":
            queues.extend(sorted(group.queue_pred.unique().tolist()))
        elif t == "odme":
            pathflow.extend([group.path_flow.min(), group.path_flow.max()])

assert total_rows == 6_980_503, total_rows
assert nulls == 0, f"{nulls} nulls"
assert set(queues) <= {0, 1}, queues
assert min(speeds) >= 0 and min(flows) >= 0 and min(pathflow) >= 0
print()
print(f"rows {total_rows:,} == 6,980,503 | nulls {nulls}")
print("task row counts:", {k: f"{v:,}" for k, v in sorted(tasks_counts.items())})
print(f"speed range [{min(speeds):.1f}, {max(speeds):.1f}] km/h")
print(f"flow range [{min(flows):.1f}, {max(flows):.1f}] veh/h")
print(f"path_flow range [{min(pathflow):.1f}, {max(pathflow):.1f}]")
print(f"validation took {time.time()-t0:.0f}s")
print("SUBMISSION FILE READY:", out_path)
"""))


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
    path = Path(__file__).resolve().parent / "tf_v2_first_submission.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} with {len(cells)} cells")


if __name__ == "__main__":
    main()
