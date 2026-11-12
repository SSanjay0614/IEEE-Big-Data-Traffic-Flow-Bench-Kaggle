"""Builds tf_v11_task3_density.ipynb from v10 (one added post-processing step).

v10 = v8 + spatial closing. This adds the Task-3 correction documented in the
public notebook `honghanhhh/trafficflowbench-lb-0-87141`, which reports it as
worth about +0.006 on the public leaderboard.

Why it works, in the author's words: the Task-3 scorer derives density from the
Task-1 answer, N = (q/v)*L, and scores the conservation identity
S_LWR = 1 - sum|dN - rhs| / sum|rhs|. The organizer boundary flows are projected
onto the released observations, so the true field satisfies the identity
exactly and a transition only loses points through the error of N at *hidden*
cells. Density changes by well under 1% per 5-minute step, so interpolating N
between the released neighbours on either side of a gap pins it down far more
tightly than q/v from a state model does.

The correction moves the predicted (q,v) towards the interpolated density in log
space: delta = log(k_interp) - log(q/v), then

    q <- q * exp(alpha * w * delta)
    v <- v * exp(-(1-alpha) * w * delta)

Settings: w = 0.5, alpha = 0.5, max_run = 2. Deliberately conservative relative
to the public grid (w up to 1.0, max_run up to 999): only cells with at most two
consecutive hidden steps are touched, where interpolation has anchors on both
sides, and the move is half-way. The public notebook selects w/alpha/max_run by
estimating the leaderboard gain on a held-out February, which we cannot
reproduce inside this kernel, so the conservative corner is used instead of a
guessed optimum.

Single change vs v10: this cell, inserted after state inference and before the
submission join.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v10" / "tf_v10_spatial_close.ipynb"
DST = HERE / "tf_v11_task3_density.ipynb"

CELL = '''\
if RUN_INFERENCE:
    # Task 3 density correction, ported from the public notebook
    # honghanhhh/trafficflowbench-lb-0-87141 (documented as about +0.006 LB).
    import state_model as sm

    CFG_T3 = dict(w=0.5, alpha=0.5, max_run=2)

    def link_geometry(p, name):
        net = DATA / "corridors" / name / "network"
        links = pd.read_csv(net / "links.csv"); links["link_id"] = links.link_id.astype(str)
        L = pd.to_numeric(links.drop_duplicates("link_id").set_index("link_id").length_km,
                          errors="coerce").reindex(p.links).to_numpy(float)
        L = np.where(np.isfinite(L), L, 1.0)
        fd = pd.read_csv(net / "fd_parameters.csv"); fd["link_id"] = fd.link_id.astype(str)
        have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in fd]
        m = fd.groupby("link_id")[have].mean().reindex(p.links)
        lanes = m.lanes.fillna(1.0).clip(lower=1.0).to_numpy(float)
        vf = m.free_speed_kmh.to_numpy(float); cap = m.capacity_vph.to_numpy(float)
        kcrit = cap / np.maximum(vf, 1)
        kjam = (m.k_jam if "k_jam" in m else pd.Series(np.nan, index=m.index)).to_numpy(float)
        kjam = np.where(np.isfinite(kjam), kjam, 100.0 * lanes)
        kjam = np.maximum(kjam, 1.05 * kcrit)
        return dict(L=L, lanes=lanes, vf=vf, cap=cap, kcrit=kcrit, kjam=kjam)

    def interp_N(v_obs, q_obs, e, L, rows, cols):
        ok = e & np.isfinite(v_obs) & np.isfinite(q_obs)
        N = np.where(ok, q_obs / np.maximum(v_obs, 1.0) * L[None, :], np.nan)
        T = N.shape[0]; pos = np.arange(T)[:, None]
        prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
        nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
        p, n = prev[rows, cols], nxt[rows, cols]
        has = (p >= 0) & (n < T); pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
        t = (rows - pp) / np.maximum(nn - pp, 1)
        Ni = N[pp, cols] * (1 - t) + N[nn, cols] * t
        return np.where(has, Ni, np.nan), np.where(has, nn - pp - 1, 10**6)

    def adjust(v_hat, q_hat, Ni, run, L_cells, cfg):
        k_hat = q_hat / np.maximum(v_hat, 1.0); k_i = Ni / L_cells
        ok = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run <= cfg["max_run"])
        d = np.where(ok, np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9)), 0.0) * cfg["w"]
        return (np.clip(v_hat * np.exp(-(1 - cfg["alpha"]) * d), 1, 180),
                np.clip(q_hat * np.exp(cfg["alpha"] * d), 0, 150000))

    state_csv = OUTPUT / "state.csv"
    state_frame = pd.read_csv(state_csv, dtype={"panel": str, "timestamp": str,
                                                "station_id": str, "link_id": str,
                                                "mask_regime": str})
    changed = 0
    anchored_cells = 0
    for panel_no, name in enumerate(sm.all_panels()):
        panel = sm.Panel(name, panel_no)
        geo = link_geometry(panel, name)
        in_panel = state_frame.index[state_frame.panel == name]
        if len(in_panel) == 0:
            continue
        for split in ("validation", "private"):
            frame = sm.month_frame(name, "", split=split)
            frame["timestamp"] = frame.timestamp.astype(str)
            times, v_obs, q_obs, eligible = sm.matrices(frame, panel)
            part = state_frame.loc[in_panel]
            row_index = pd.Index(times).get_indexer(part.timestamp)
            keep = row_index >= 0
            if not keep.any():
                continue
            rows = row_index[keep]
            idx = part.index.values[keep]
            cols = pd.Index(panel.links).get_indexer(part.link_id.values[keep])
            valid = cols >= 0
            rows, idx, cols = rows[valid], idx[valid], cols[valid]
            Ni, run = interp_N(v_obs, q_obs, eligible, geo["L"], rows, cols)
            v_hat = state_frame.speed_kmh.values[idx].astype(float)
            q_hat = state_frame.flow_vph.values[idx].astype(float)
            v_new, q_new = adjust(v_hat, q_hat, Ni, run, geo["L"][cols], CFG_T3)
            changed += int((np.abs(v_new - v_hat) > 1e-6).sum())
            anchored_cells += int(np.isfinite(Ni).sum())
            state_frame.loc[idx, "speed_kmh"] = v_new
            state_frame.loc[idx, "flow_vph"] = q_new
            print(name, split, f"{len(idx):,} cells, {np.isfinite(Ni).mean():.0%} anchored")
    state_frame.to_csv(state_csv, index=False)
    print(f"task3 density correction {CFG_T3}: anchored {anchored_cells:,}, changed {changed:,} state cells")
else:
    print("Inference disabled.")
'''

nb = json.loads(SRC.read_text(encoding="utf-8"))

# insert after the state-inference cell (the one calling predict_states)
insert_at = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "predict_states(state_predictor)" in "".join(cell["source"]):
        insert_at = i + 1
assert insert_at is not None, "state inference cell not found"

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": CELL.splitlines(keepends=True),
    "outputs": [],
    "execution_count": None,
}
nb["cells"].insert(insert_at, new_cell)

nb["cells"][0]["source"] = [line + "\n" for line in """\
# TF V11 — v10 + Task-3 density correction

v10 scores are pending. This adds the Task-3 correction from the public notebook
`honghanhhh/trafficflowbench-lb-0-87141` (documented there as about **+0.006**),
applied as a post-processing step on our own state predictions.

Task 3 scores the conservation identity on density, N = (q/v)*L. The organizer
flows are projected onto the released observations, so the identity holds
exactly in truth and only hidden cells contribute error. Density moves by well
under 1% per 5-minute step, which makes the interpolated value between the
released neighbours on either side of a gap far more accurate than a state
model's q/v.

Settings are conservative (w = 0.5, alpha = 0.5, only cells with at most two
consecutive hidden steps). The public notebook picks its settings by estimating
the leaderboard gain on held-out February, which this kernel cannot do.

Everything else -- state model, queue (v10 closing), ODME, physics clip -- is
unchanged from v10.
""".splitlines(keepends=True)]

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
        compile(src, "cell", "exec")
print(f"wrote {DST.name}; inserted at {insert_at}; code cells compile")
