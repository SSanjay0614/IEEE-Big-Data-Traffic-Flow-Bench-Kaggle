"""Builds kernel v7 = v3 pipeline with the corrected Task-2 onset fill.

Evidence (kernel v6, offline, labelled February, official block IoU, identical windows):

  onset policy                 IoU     note
  last_only @0.55 (shipped)   0.370    step 6 only, steps 1-5 zeroed
  all_steps @0.55             0.462    threshold the whole block
  fill_from_4 @0.40           0.532    step-6 links held over steps 4,5,6
  fill_from_4 @0.30           0.536    best of the grid

fill_from_4 beat the shipped rule in every one of the 8 scored panels (+0.13 to +0.20).
The reason is physical: truth positives per step ramp [0.54, 1.09, 1.66, 2.35, 2.97, 3.78],
so a queue that is present at step 6 was already forming at steps 4-5. A single frame at
step 6 under-covers the block, and IoU charges for the miss.

Threshold: 0.40. The onset curve is flat between 0.30 and 0.55 here (0.536 -> 0.522), so this
is not a tuned edge; it sits between our local optimum and the author's validated 0.55.

Queue is 30% of the score with onset/ongoing weighted equally, so the measured delta
projects to 0.30 * 0.5 * 0.16 ~= +0.024 overall, if it transfers to the hidden windows.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "kernel_v3" / "tf_v3_public_pipeline.ipynb"

ONSET_THRESHOLD = 0.40
ONSET_FILL_FROM = 4          # first horizon step that carries the mask (steps 4,5,6)

nb = json.loads(V3.read_text(encoding="utf-8"))
cells = list(nb["cells"])

OLD_SETTINGS = '    onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}'
NEW_SETTINGS = (
    '    # v7: onset mask is held over the last three horizon steps instead of only step 6\n'
    f'    onset_settings = {{"gpu_weight": 0.5, "threshold": {ONSET_THRESHOLD}, "fill_from": {ONSET_FILL_FROM}}}'
)
OLD_FILL = """                prediction = np.zeros((HORIZON, len(panel.links)), dtype=np.uint8)
                prediction[-1] = selected"""
NEW_FILL = f"""                prediction = np.zeros((HORIZON, len(panel.links)), dtype=np.uint8)
                prediction[{ONSET_FILL_FROM} - 1:] = selected[None, :]"""

patched = 0
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if OLD_SETTINGS not in src:
        continue
    src = src.replace(OLD_SETTINGS, NEW_SETTINGS)
    assert OLD_FILL in src, "onset fill statement not found"
    src = src.replace(OLD_FILL, NEW_FILL)
    c["source"] = src.splitlines(keepends=True)
    patched += 1
assert patched == 1, f"expected to patch 1 cell, patched {patched}"

# replace the reproduction-notes cell with an honest account of what changed
cells[-1] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## What v7 changes\n",
        "\n",
        "One line of Task 2. The pipeline selected onset links from a single probability vector and\n",
        "wrote that mask at the final horizon step only, leaving steps 1-5 at zero. Measured against\n",
        "the official space-time IoU on labelled February windows (kernel v6), that rule scores\n",
        "**0.370**, while holding the same mask over steps 4-6 scores **0.532** - better in all eight\n",
        "scored panels. Truth positives per step ramp `[0.54, 1.09, 1.66, 2.35, 2.97, 3.78]`, so a\n",
        "queue present at step 6 has already been forming earlier; a single frame under-covers.\n",
        "\n",
        "Threshold moves 0.55 -> 0.40. The curve is flat there (0.536 at 0.30, 0.522 at 0.55), so\n",
        "this is a plateau choice, not an edge fit.\n",
        "\n",
        "Everything else - state reconstruction, ongoing queue threshold 0.40, ODME - is unchanged\n",
        "from the published pipeline, so this submission isolates the queue change.\n",
    ],
}

out = dict(nb)
out["cells"] = cells
path = HERE / "tf_v7_onset_fill.ipynb"
path.write_text(json.dumps(out, indent=1), encoding="utf-8")

for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("%%"):
        continue
    compile(src, "cell", "exec")
print(f"wrote {path.name}: {len(cells)} cells; plain code cells compile")
print(f"onset threshold {ONSET_THRESHOLD}, fill from step {ONSET_FILL_FROM}")

meta = json.loads((V3.parent / "kernel-metadata.json").read_text())
meta.update({"id": "sanjay0614/tf-v7-onset-fill", "title": "TF V7 Onset Fill", "code_file": path.name})
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote kernel-metadata.json")
