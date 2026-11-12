"""Builds tf_v9_onset_070.ipynb from the v8 notebook.

v8 (onset threshold 0.55, step 6 only) measured 0.86515 against v4's 0.85457 with
threshold 0.40 -- a clean one-knob A/B worth +0.0705 per unit of threshold, and
the second independent confirmation that onset false positives are the expensive
error on the official window population.

v9 takes one more step in that direction: threshold 0.70, everything else
byte-identical. The purpose is to locate the knee of the curve with the last
submission of the day. If 0.70 still improves, the curve is flat-topped and the
next slots can scan 0.85 and 1.0; if it drops, 0.55 is the optimum and that is
settled for good.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v8" / "tf_v8_onset_revert.ipynb"
DST = HERE / "tf_v9_onset_070.ipynb"

OLD = 'onset_settings = {"gpu_weight": 0.5, "threshold": 0.55}'
NEW = 'onset_settings = {"gpu_weight": 0.5, "threshold": 0.70}'

nb = json.loads(SRC.read_text(encoding="utf-8"))

hits = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if OLD in source:
        cell["source"] = [l + "\n" for l in source.replace(OLD, NEW).splitlines()]
        hits += 1
assert hits == 1, f"expected exactly one occurrence of the onset threshold, found {hits}"

inf = "".join(nb["cells"][25]["source"])
assert '"threshold": 0.70' in inf and 'prediction[-1] = selected' in inf
assert "selected[None, :]" not in inf, "fill must not be present"
compile(inf, "queue_inference", "exec")

nb["cells"][0]["source"] = [
    line + "\n"
    for line in """# TF V9 \u2014 onset threshold 0.70 (step 6 only)

v8 measured **0.86515** with the onset threshold at 0.55, against **0.85457** at 0.40. Same file
apart from that one number, so the curve is real: on the official window population, extra onset
false positives are the expensive error.

v9 moves one step further to **0.70** and changes nothing else. This locates the knee of the
curve using the last submission of the day.
""".splitlines()
]
nb["cells"][0]["cell_type"] = "markdown"

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {DST.name}; threshold patched once; inference cell compiles")

meta = {
    "id": "sanjay0614/tf-v9-onset-070",
    "title": "TF V9 Onset 070",
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
