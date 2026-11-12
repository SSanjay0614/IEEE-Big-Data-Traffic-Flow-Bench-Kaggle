"""Builds tf_v13_queue_open.ipynb from v8 (the 0.86515 configuration).

v10 added a 1-D closing to the ongoing queue -- an interior dry link between two wet
neighbours was flipped wet -- and scored 0.86143, a loss of 0.0037 for only 379 extra
positive cells. That says the queue term is precision-sensitive at our operating
point: extra positives are expensive, so the mask is not under-filled.

This is the mirror experiment on the same mask: drop a positive link whose two
corridor neighbours are both dry. A freeway queue shorter than one link is rare,
so a lone positive is more likely a false positive than a one-link queue. Cost is
a handful of cells, so even a null result is cheap to learn.

Applied to the ongoing mask only. The onset mask is left byte-identical: onset
windows are constructed to contain a queue and our fallback writes one link, so an
opening there would delete the only positive it has.

Everything else -- state, ODME, onset at 0.55 with argmax fallback, physics clip --
is byte-identical to v8.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "kernel_v8" / "tf_v8_onset_revert.ipynb"
DST = HERE / "tf_v13_queue_open.ipynb"

OLD = (
    '                probability = ongoing_model.predict(X, num_threads=THREADS)\n'
    '                prediction = (probability >= 0.4000000000000001).reshape(HORIZON, -1).astype(np.uint8)\n'
)
NEW = (
    '                probability = ongoing_model.predict(X, num_threads=THREADS)\n'
    '                prediction = (probability >= 0.4000000000000001).reshape(HORIZON, -1).astype(np.uint8)\n'
    '                # v13: 1-D morphological opening along the corridor. v10 showed that 379 extra\n'
    '                # positives cost 0.0037, so the term is precision-sensitive; a lone wet link with\n'
    '                # both neighbours dry is the cheapest false positive to remove. Onset untouched.\n'
    '                for _h in range(HORIZON):\n'
    '                    row = prediction[_h].copy()\n'
    '                    if row.size > 2:\n'
    '                        prediction[_h, 1:-1] &= ~((row[:-2] == 0) & (row[2:] == 0))\n'
)

PROBE = (
    '    queue = pd.concat(queue_parts, ignore_index=True)\n'
    '    queue.to_csv(OUTPUT / "queue.csv", index=False)\n'
)
PROBE_NEW = (
    '    queue = pd.concat(queue_parts, ignore_index=True)\n'
    '    queue.to_csv(OUTPUT / "queue.csv", index=False)\n'
    '    print("v13 opening audit: cells", len(queue), "positives", int(queue.queue_pred.sum()))\n'
)

nb = json.loads(SRC.read_text(encoding="utf-8"))
applied = {"open": 0, "audit": 0}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    out = src
    if OLD in out:
        out = out.replace(OLD, NEW)
        applied["open"] += 1
    if PROBE in out:
        out = out.replace(PROBE, PROBE_NEW)
        applied["audit"] += 1
    if out != src:
        cell["source"] = [line + "\n" for line in out.splitlines()]

assert applied == {"open": 1, "audit": 1}, f"patch counts wrong: {applied}"

nb["cells"][0]["source"] = [line + "\n" for line in """\
# TF V13 — morphological opening on the ongoing queue

v8 scored **0.86515**. v10 added a closing (fill a dry link between two wet ones) and scored
**0.86143**: 379 extra positive cells cost 0.0037. So extra positives are expensive here and the
mask is not under-filled.

This is the mirror test: drop a positive link whose corridor neighbours are both dry. A queue
shorter than one link is rare, so a lone positive is more likely a false positive. The onset mask is
untouched -- onset windows contain a queue by construction and our fallback writes exactly one link,
so an opening there would delete the only positive.

State, ODME, onset settings and the physics clip are byte-identical to v8, so the delta against
0.86515 is the opening.
""".splitlines(keepends=True)]

DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and not src.lstrip().startswith("%%"):
        compile(src, "cell", "exec")
print(f"wrote {DST.name}; patches {applied}; code cells compile")
