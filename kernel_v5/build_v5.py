"""Builds kernel v5 = proven v3 pipeline + honghanhhh's Task-3 density post-processor.

Why: the 0.86517 pipeline's own notes say the displayed score is historical, not a
re-measurement, so our 0.85457 rerun is the honest base. The remaining public lever is
Task 3 (15% of the score), which is scored on the Task 1 answers: density N = q/v * L must
satisfy vehicle conservation. honghanhhh's post-processor interpolates N between released
time neighbours (which pins it far more tightly than the state models can) and only applies
the correction when a local held-out grid says the estimated LB gain exceeds 0.001.

Inputs it needs, all already present in the pipeline notebook:
  * competition data          -> DATA   (submission_key.csv)
  * state checkpoints         -> ASSETS (speed_d8.cbm ...)
  * tfb_source/ modules       -> SOURCE (state_model, prepare_state, state_inference)
  * a base submission         -> /kaggle/working/submission.csv written by the cells above

The base file is copied to base_submission.csv first so a mid-run failure cannot destroy it.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "kernel_v3" / "tf_v3_public_pipeline.ipynb"
PUB = HERE.parent.parent / "scratch" / "comps" / "traffic_flow" / "public_kernels"
POST = PUB / "pull_test" / "trafficflowbench-lb-0-87141.ipynb"

nb = json.loads(V3.read_text(encoding="utf-8"))
post = json.loads(POST.read_text(encoding="utf-8"))
cells = list(nb["cells"])

# The post-processor's own setup cell re-derives DATA/ASSETS/SRC and re-copies the modules.
# We already have all three in scope from the pipeline, so replace that cell with an adapter.
adapter = '''\
import os, sys, glob, shutil, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

CFG = dict(
    base_sub=None,      # None -> the submission.csv the pipeline cells just wrote
    days=7,             # February days per panel used for the local evaluation
    min_gain=0.001,     # required estimated LB gain before the submission is changed
    force=None,         # e.g. dict(w=1.0, alpha=1.0, max_run=2) to skip auto-selection
    grid_w=(0.25, 0.5, 0.75, 1.0), grid_alpha=(0.0, 0.25, 0.5, 0.75, 1.0), grid_run=(1, 2, 4, 999),
)
T0 = time.time(); elapsed = lambda: f"{time.time() - T0:.0f}s"

# Already resolved by the pipeline cells above:
#   DATA    competition data dir      ASSETS  checkpoint dir      SOURCE  tfb_source/ modules
DATA, ASSETS, SOURCE = Path(DATA), Path(ASSETS), Path(SOURCE)
BASE_SUB = CFG["base_sub"] or str(Path("/kaggle/working/submission.csv"))
assert Path(BASE_SUB).is_file(), f"base submission missing: {BASE_SUB}"
BASE_COPY = Path("/kaggle/working/base_submission.csv")
shutil.copyfile(BASE_SUB, BASE_COPY)          # the post-processor rewrites submission.csv
print("data:", DATA); print("checkpoints:", ASSETS); print("source:", SOURCE)
print("BASE submission:", BASE_SUB, f"({Path(BASE_SUB).stat().st_size/1e6:.0f} MB)")

sys.path.insert(0, str(SOURCE))
import state_model as sm
from prepare_state import features
from state_inference import StatePredictor
predictor = StatePredictor(ASSETS)
PANELS = sm.all_panels()
print("panels:", PANELS, elapsed())
'''

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Task 3 density post-processing (gated)\n",
        "\n",
        "Task 3 is 15% of the score and is computed from the Task 1 answers: the scorer forms "
        "`N = (q/v) * L` and checks `N[t+1] - N[t] == rhs[t]` on 5-minute transitions. The "
        "released neighbours in time pin `N` to well under 1% per step, so interpolating `N` "
        "between them can beat `q_hat / v_hat`.\n",
        "\n",
        "Step 1 measures the effect on a February hold-out with the same state checkpoints. "
        "Step 2 grids over the correction (strength `w`, split `alpha` between flow and speed, "
        "max hidden run). Step 3 applies it **only if** the estimated combined gain exceeds "
        "0.001 - otherwise the base submission is left untouched and nothing is submitted.\n",
        "\n",
        "Method: credit to honghanhhh, `trafficflowbench-lb-0-87141`.\n",
    ],
})
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": adapter.splitlines(keepends=True)})

# Post-processor body: their cells 3..10 (skip cell 2, a markdown lead-in).
for c in post["cells"][3:]:
    cells.append(json.loads(json.dumps(c)))

out_nb = dict(nb)
out_nb["cells"] = cells
path = HERE / "tf_v5_task3_density.ipynb"
path.write_text(json.dumps(out_nb, indent=1), encoding="utf-8")

# Compile check on all code cells.
code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
# compile per cell, skipping %%writefile cells (their bodies are whole modules, checked on Kaggle)
_checked = 0
for c in cells:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("%%"):
        continue
    compile(src, "cell", "exec")
    _checked += 1
print(f"syntax check: {_checked} plain code cells compile")

print(f"wrote {path.name}: {len(cells)} cells (v3 had {len(nb['cells'])}), compiles OK")

meta = json.loads((V3.parent / "kernel-metadata.json").read_text())
meta.update({
    "id": "sanjay0614/tf-v5-task3-density",
    "title": "TF V5 Task3 Density",
    "code_file": path.name,
})
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote kernel-metadata.json")

