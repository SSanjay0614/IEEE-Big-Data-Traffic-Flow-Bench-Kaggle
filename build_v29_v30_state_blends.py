import csv
import math
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "kernel_v15" / "cloud_output" / "submission.csv"
STATE_SOURCE = ROOT / "kernel_v22" / "cloud_output" / "submission.csv"
EXPECTED_ROWS = 6_980_503
EXPECTED_TASK_ROWS = {"state": 6_735_795, "queue": 174_000, "odme": 70_708}
SENTINEL = object()
BLENDS = (
    (29, 0.75),
    (30, 0.90),
)


def build(version: int, state_weight: float) -> None:
    output = ROOT / f"kernel_v{version}_state_blend" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {task: 0 for task in EXPECTED_TASK_ROWS}
    blended_rows = 0

    with (
        BASE.open(newline="", encoding="utf-8") as base_file,
        STATE_SOURCE.open(newline="", encoding="utf-8") as state_file,
        output.open("w", newline="", encoding="utf-8") as output_file,
    ):
        base_reader, state_reader = csv.reader(base_file), csv.reader(state_file)
        base_header, state_header = next(base_reader), next(state_reader)
        if base_header != state_header:
            raise ValueError("V15 and V22 submission headers differ")
        columns = {name: index for index, name in enumerate(base_header)}
        writer = csv.writer(output_file)
        writer.writerow(base_header)

        for row_number, rows in enumerate(
            zip_longest(base_reader, state_reader, fillvalue=SENTINEL), start=1
        ):
            base_row, state_row = rows
            if base_row is SENTINEL or state_row is SENTINEL:
                raise ValueError(f"V15/V22 row counts differ at data row {row_number}")
            if len(base_row) != len(base_header) or len(state_row) != len(state_header):
                raise ValueError(f"Invalid column count at data row {row_number}")
            if (
                base_row[columns["submission_id"]] != state_row[columns["submission_id"]]
                or base_row[columns["task"]] != state_row[columns["task"]]
            ):
                raise ValueError(f"V15/V22 row alignment differs at data row {row_number}")

            task = base_row[columns["task"]]
            if task not in counts:
                raise ValueError(f"Unexpected task {task!r} at data row {row_number}")
            counts[task] += 1
            result = base_row.copy()
            if task == "state":
                for name in ("speed_kmh", "flow_vph"):
                    base_value = float(base_row[columns[name]])
                    state_value = float(state_row[columns[name]])
                    value = base_value + state_weight * (state_value - base_value)
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {name} at data row {row_number}")
                    if value != base_value:
                        blended_rows += 1
                    result[columns[name]] = format(value, ".15g")
            elif task == "queue":
                if base_row[columns["queue_pred"]] not in ("0", "0.0", "1", "1.0"):
                    raise ValueError(f"Invalid queue label at data row {row_number}")
            elif (
                not math.isfinite(float(base_row[columns["path_flow"]]))
                or float(base_row[columns["path_flow"]]) < 0
            ):
                raise ValueError(f"Invalid path flow at data row {row_number}")
            if any(value == "" for value in result):
                raise ValueError(f"Empty submission field at data row {row_number}")
            writer.writerow(result)

    if sum(counts.values()) != EXPECTED_ROWS or counts != EXPECTED_TASK_ROWS:
        raise ValueError(f"Unexpected task row counts: {counts}")
    if blended_rows == 0:
        raise ValueError("The requested blend did not change any state values")
    print(
        f"V{version}: alpha={state_weight:.2f}; {EXPECTED_ROWS:,} rows; "
        f"task rows={counts}; blended state values={blended_rows:,}; output={output}"
    )


if __name__ == "__main__":
    for candidate_version, weight in BLENDS:
        build(candidate_version, weight)
