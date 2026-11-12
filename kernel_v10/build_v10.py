"""Builds tf_v10_spatial_close.ipynb from the v8 notebook (single change).

v8 (0.86515) predicts the ongoing queue as `probability >= 0.40` per link-step
and nothing else. The public "top-rank" notebook additionally applies a 1-D
morphological closing along each corridor: an interior link predicted as 0 is
flipped to 1 when both of its corridor neighbours are 1. The physical
justification is that a queue is contiguous along a freeway, so a single dry
link inside a queue is almost certainly a false negative, while flipping it
costs nothing when the window is empty.

That is the whole change. Onset (step 6 only, threshold 0.55), state, ODME and
the physics clip are identical to v8, so the leaderboard delta against 0.86515
attributes to closing alone.

Deliberately NOT included, to keep the attribution clean:
  - the public notebook's onset threshold 0.52 and top-k 2 fallback. We measured
    0.40 -> 0.85457, 0.55 -> 0.86515, 0.70 -> 0.86206, so 0.55 is a measured
    interior maximum and 0.52 is untested but close. top-k 2 adds positives,
    which is the direction that cost us 0.0675 in v7.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v8" / "tf_v8_onset_revert.ipynb"
DST = HERE / "tf_v10_spatial_close.ipynb"

OLD = (
    '                probability = ongoing_model.predict(X, num_threads=THREADS)\n'
    '                prediction = (probability >= 0.4000000000000001).reshape(HORIZON, -1).astype(np.uint8)\n'
)
NEW = (
    '                probability = ongoing_model.predict(X, num_threads=THREADS)\n'
    '                prediction = (probability >= 0.4000000000000001).reshape(HORIZON, -1).astype(np.uint8)\n'
    '                # v10: 1-D morphological closing along the corridor. A queue is contiguous in\n'
    '                # space, so a single link predicted dry between two wet neighbours is a recall\n'
    '                # miss that costs two cells (that link at that step, plus the split it causes).\n'
    '                # Filling it only adds a cell where the truth is already 1 almost everywhere.\n'
    '                for _h in range(HORIZON):\n'
    '                    row = prediction[_h].copy()\n'
    '                    if row.size > 2:\n'
    '                        prediction[_h, 1:-1] |= ((row[:-2] == 1) & (row[2:] == 1))\n'
)

PROBE = (
    '    queue = pd.concat(queue_parts, ignore_index=True)\n'
    '    queue.to_csv(OUTPUT / "queue.csv", index=False)\n'
)
PROBE_NEW = (
    '    queue = pd.concat(queue_parts, ignore_index=True)\n'
    '    queue.to_csv(OUTPUT / "queue.csv", index=False)\n'
    '    _closed = (queue["queue_pred"] == 0)\n'
    '    print("v10 closing audit: cells", len(queue), "positives", int(queue.queue_pred.sum()))\n'
)

nb = json.loads(SRC.read_text(encoding="utf-8"))
applied = {"close": 0, "audit": 0}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    out = src
    if OLD in out:
        out = out.replace(OLD, NEW)
        applied["close"] += 1
    if PROBE in out:
        out = out.replace(PROBE, PROBE_NEW)
        applied["audit"] += 1
    if out != src:
        cell["source"] = [line + "\n" for line in out.splitlines()]

assert applied == {"close": 1, "audit": 1}, f"patch counts wrong: {applied}"

md = nb["cells"][0]
md["source"] = [line + "\n" for line in """\
# TF V10 — spatial morphological closing on the ongoing queue

v8 scored **0.86515**. This is v8 with exactly one added step: after the ongoing
mask is thresholded at 0.40, an interior link predicted 0 flips to 1 when both of
its corridor neighbours at the same step are 1.

Reason: a queue occupies a contiguous run of links along a corridor. A lone dry
link inside a wet run is a false negative that costs that cell and splits the
prediction, so closing is nearly free in precision terms and recovers recall
where the model is least confident (the middle of a queue).

Onset, state, ODME and the physics clip are byte-identical to v8, so any delta
against 0.86515 is the closing.
""".splitlines(keepends=True)]

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
        compile(src, "cell", "exec")
print(f"wrote {DST.name}; patches {applied}; code cells compile")
