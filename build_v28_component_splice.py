import csv
import math
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "kernel_v15" / "cloud_output" / "submission.csv"
STATE_SOURCE = ROOT / "kernel_v22" / "cloud_output" / "submission.csv"
OUTPUT = ROOT / "kernel_v28_component_splice" / "submission.csv"
EXPECTED_ROWS = 6_980_503
EXPECTED_TASK_ROWS = {"state": 6_735_795, "queue": 174_000, "odme": 70_708}
SENTINEL = object()


def main() -> None:
    for path in (BASE, STATE_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    counts = {task: 0 for task in EXPECTED_TASK_ROWS}
    changed_state_rows = 0

    with (
        BASE.open(newline="", encoding="utf-8") as base_file,
        STATE_SOURCE.open(newline="", encoding="utf-8") as state_file,
        OUTPUT.open("w", newline="", encoding="utf-8") as output_file,
    ):
        base_reader, state_reader = csv.reader(base_file), csv.reader(state_file)
        base_header, state_header = next(base_reader), next(state_reader)
        if base_header != state_header:
            raise ValueError("V15 and V22 submission headers differ")
        columns = {name: index for index, name in enumerate(base_header)}
        required = ("submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"Missing submission columns: {missing}")

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
                    value = state_row[columns[name]]
                    if not value:
                        raise ValueError(f"Empty {name} at data row {row_number}")
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Non-finite {name} at data row {row_number}")
                    if value != base_row[columns[name]]:
                        changed_state_rows += 1
                    result[columns[name]] = value
            elif task == "queue":
                queue = base_row[columns["queue_pred"]]
                if queue not in ("0", "0.0", "1", "1.0"):
                    raise ValueError(f"Invalid queue label at data row {row_number}: {queue!r}")
            else:
                path_flow = base_row[columns["path_flow"]]
                if not math.isfinite(float(path_flow)) or float(path_flow) < 0:
                    raise ValueError(f"Invalid path flow at data row {row_number}")
            if any(value == "" for value in result):
                raise ValueError(f"Empty submission field at data row {row_number}")
            writer.writerow(result)

    if sum(counts.values()) != EXPECTED_ROWS or counts != EXPECTED_TASK_ROWS:
        raise ValueError(f"Unexpected task row counts: {counts}")
    if changed_state_rows == 0:
        raise ValueError("V22 state output did not change any V15 state predictions")
    print(
        f"Wrote {EXPECTED_ROWS:,} rows; task rows={counts}; "
        f"V22 state values differ on {changed_state_rows:,} cells."
    )
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
