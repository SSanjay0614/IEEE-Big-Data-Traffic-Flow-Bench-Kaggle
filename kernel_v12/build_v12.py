"""Builds tf_v12_onset_topk.ipynb from v10 (one Task-2 change).

v10 = v8 (0.86515) + spatial closing on the ongoing queue. This adds the only
Task-2 settings the public "top-rank" notebook has that v10 does not:

  threshold 0.55 -> 0.52,  fallback argmax(1 link) -> top-2 links

Both come from `sujanmajhisuzan/trafficflowbench-top-rank-pipeline-0-90-target`,
whose author calls 0.52 the F1-optimal threshold and top-2 a "guaranteed onset
fallback" because every onset window contains a queue by construction.

Our own measurements bracket 0.55 as the interior maximum (0.40 -> 0.85457,
0.55 -> 0.86515, 0.70 -> 0.86206), so 0.52 is a small move towards extra recall
on a plateau, not an edge fit. The fallback change only fires when nothing
clears the threshold, which is the case where we currently predict exactly one
link and the truth is more likely to be a run of two.

Everything else -- state, ODME, ongoing threshold 0.40, v10 closing, physics
clip -- is byte-identical to v10, so the delta against v10's score is this pair
of onset settings.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v10" / "tf_v10_spatial_close.ipynb"
DST = HERE / "tf_v12_onset_topk.ipynb"

OLD = (
    '    onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}\n'
)
NEW = (
    '    # v12: public top-rank settings -- threshold 0.52 (their F1-optimal point) and a\n'
    '    # two-link fallback, since every onset window contains a queue by construction.\n'
    '    onset_settings = {"gpu_weight": 0.5, "threshold": 0.52, "top_k": 2}\n'
)

OLD2 = (
    '                if not selected.any():\n'
    '                    selected[np.argmax(probability)] = True\n'
)
NEW2 = (
    '                if selected.sum() < onset_settings["top_k"]:\n'
    '                    order = np.argsort(probability)[-onset_settings["top_k"]:]\n'
    '                    selected[order] = True\n'
)

nb = json.loads(SRC.read_text(encoding="utf-8"))
applied = {"settings": 0, "fallback": 0}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    out = src
    if OLD in out:
        out = out.replace(OLD, NEW)
        applied["settings"] += 1
    if OLD2 in out:
        out = out.replace(OLD2, NEW2)
        applied["fallback"] += 1
    if out != src:
        cell["source"] = [line + "\n" for line in out.splitlines()]

assert applied == {"settings": 1, "fallback": 1}, f"patch counts wrong: {applied}"

nb["cells"][0]["source"] = [line + "\n" for line in """\
# TF V12 — onset threshold 0.52 and a two-link fallback

v10 = v8 (0.86515) + spatial closing. This is v10 plus the two Task-2 settings the
public top-rank notebook has that we do not: **onset threshold 0.52** (their
F1-optimal point; we measured 0.55 as the interior maximum of 0.40/0.55/0.70) and
a **top-2 fallback** when nothing clears the threshold, instead of our argmax-1.

Why the fallback matters: every onset window contains a queue by construction, so
predicting a single link when the probabilities are all low is a structural
under-prediction of a run. Two links is the smallest run that can still be scored
as a queue.

Ongoing queue threshold 0.40, v10 closing, state, ODME and the physics clip are
unchanged, so the leaderboard delta against v10 is this pair of onset settings.
""".splitlines(keepends=True)]

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
        compile(src, "cell", "exec")
print(f"wrote {DST.name}; patches {applied}; code cells compile")
