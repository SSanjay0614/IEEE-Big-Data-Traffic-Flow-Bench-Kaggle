"""Builds tf_v8_onset_revert.ipynb from the v7 notebook.

v7 filled the onset mask over horizon steps 4-6 and scored 0.78710 against v4's
0.85457. The official Task 2 scorer is per-window: IoU over every eligible
link-step cell in the window, averaged over windows. Filling steps 4-5 put false
positives at two extra steps in every window whose queue actually arrives at
step 6 -- which is what the official onset windows look like.

So v8 restores the shipped rule exactly:

    prediction[-1] = selected

and additionally restores the shipped onset threshold. The original pipeline
(`kernel_v3`, the 0.86517 configuration) used threshold 0.55; v4 lowered it to
0.40. Since the expensive direction for onset is extra positives, 0.55 is the
better default to test.

Nothing else changes: state model, ODME solve, ongoing threshold 0.40 and the
physics clip are untouched, so the delta against v4 isolates the onset term.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v7" / "tf_v7_onset_fill.ipynb"
DST = HERE / "tf_v8_onset_revert.ipynb"

PATCHES = [
    (
        '# v7: onset mask is held over the last three horizon steps instead of only step 6\n'
        '    onset_settings = {"gpu_weight": 0.5, "threshold": 0.4, "fill_from": 4}',
        '# v8: onset rule restored to the shipped single-step form, at the shipped threshold.\n'
        '    # v7 held the step-6 mask over steps 4-6 and lost 0.0675 on the public LB: the official\n'
        '    # scorer is per-window, and the official onset windows have their queue arriving at\n'
        '    # step 6, so filling steps 4-5 just adds false positives.\n'
        '    onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}',
    ),
    (
        'prediction[4 - 1:] = selected[None, :]',
        'prediction[-1] = selected',
    ),
]

nb = json.loads(SRC.read_text(encoding="utf-8"))

applied = {i: 0 for i in range(len(PATCHES))}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    updated = source
    for i, (old, new) in enumerate(PATCHES):
        if old in updated:
            updated = updated.replace(old, new)
            applied[i] += 1
    if updated != source:
        cell["source"] = [line + "\n" for line in updated.splitlines()]

missing = [i for i, n in applied.items() if n != 1]
assert not missing, f"patches not applied exactly once: {applied}"

# Prove the patch landed and the code still parses.
inf = "".join(nb["cells"][25]["source"])
assert '"threshold": 0.55' in inf and '"fill_from"' not in inf
assert "prediction[-1] = selected" in inf and "selected[None, :]" not in inf
compile(inf, "queue_inference", "exec")

nb["cells"][0]["source"] = [
    line + "\n"
    for line in """# TF V8 \u2014 onset reverted to step 6 at the shipped threshold

v7 held the onset mask over horizon steps 4\u20136 and measured **0.78710** against v4's **0.85457**.
The official Task 2 score is a per-window IoU over every eligible link-step cell; the official
onset windows have their queue arriving at step 6, so filling steps 4\u20135 only adds false
positives.

v8 restores `prediction[-1] = selected` and the shipped onset threshold of **0.55** (v4 had
lowered it to 0.40). State, ODME, ongoing threshold 0.40 and the physics clip are byte-identical
to v4, so the leaderboard delta isolates the onset term.
""".splitlines()
]
nb["cells"][0]["cell_type"] = "markdown"

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {DST.name} ({len(nb['cells'])} cells); all patches applied once; inference cell compiles")

meta = {
    "id": "sanjay0614/tf-v8-onset-revert",
    "title": "TF V8 Onset Revert",
    "code_file": DST.name,
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
