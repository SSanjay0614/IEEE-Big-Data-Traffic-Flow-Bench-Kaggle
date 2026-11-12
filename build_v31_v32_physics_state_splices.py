import csv
import math
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "kernel_v15" / "cloud_output" / "submission.csv"
EXPECTED_ROWS = 6_980_503
EXPECTED_TASK_ROWS = {"state": 6_735_795, "queue": 174_000, "odme": 70_708}
SENTINEL = object()
CANDIDATES = (
    (31, ROOT / "kernel_v23" / "cloud_output" / "submission.csv"),
    (32, ROOT / "kernel_v24" / "cloud_output" / "submission.csv"),
)


def build(version: int, state_source: Path) -> None:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not state_source.is_file():
        raise FileNotFoundError(state_source)
    output = ROOT / f"kernel_v{version}_physics_state_splice" / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {task: 0 for task in EXPECTED_TASK_ROWS}
    changed = 0

    with (
        BASE.open(newline="", encoding="utf-8") as base_file,
        state_source.open(newline="", encoding="utf-8") as source_file,
        output.open("w", newline="", encoding="utf-8") as output_file,
    ):
        base_reader, source_reader = csv.reader(base_file), csv.reader(source_file)
        base_header, source_header = next(base_reader), next(source_reader)
        if base_header != source_header:
            raise ValueError(f"V15 and V{version} headers differ")
        columns = {name: index for index, name in enumerate(base_header)}
        writer = csv.writer(output_file)
        writer.writerow(base_header)

        for row_number, rows in enumerate(
            zip_longest(base_reader, source_reader, fillvalue=SENTINEL), start=1
        ):
            base_row, source_row = rows
            if base_row is SENTINEL or source_row is SENTINEL:
                raise ValueError(f"Row counts differ at data row {row_number}")
            if len(base_row) != len(base_header) or len(source_row) != len(source_header):
                raise ValueError(f"Invalid column count at data row {row_number}")
            if (
                base_row[columns["submission_id"]] != source_row[columns["submission_id"]]
                or base_row[columns["task"]] != source_row[columns["task"]]
            ):
                raise ValueError(f"Row alignment differs at data row {row_number}")

            task = base_row[columns["task"]]
            if task not in counts:
                raise ValueError(f"Unexpected task {task!r}")
            counts[task] += 1
            result = base_row.copy()
            if task == "state":
                for name in ("speed_kmh", "flow_vph"):
                    value = source_row[columns[name]]
                    if not value or not math.isfinite(float(value)):
                        raise ValueError(f"Invalid {name} at data row {row_number}")
                    changed += value != base_row[columns[name]]
                    result[columns[name]] = value
            elif task == "queue":
                if base_row[columns["queue_pred"]] not in ("0", "0.0", "1", "1.0"):
                    raise ValueError(f"Invalid queue label at data row {row_number}")
            else:
                path_flow = float(base_row[columns["path_flow"]])
                if not math.isfinite(path_flow) or path_flow < 0:
                    raise ValueError(f"Invalid path flow at data row {row_number}")
            if any(value == "" for value in result):
                raise ValueError(f"Empty field at data row {row_number}")
            writer.writerow(result)

    if sum(counts.values()) != EXPECTED_ROWS or counts != EXPECTED_TASK_ROWS:
        raise ValueError(f"Unexpected task row counts: {counts}")
    if changed == 0:
        raise ValueError(f"V{version} state did not differ from V15")
    print(
        f"V{version}: source={state_source}; rows={EXPECTED_ROWS:,}; "
        f"task rows={counts}; changed state values={changed:,}; output={output}"
    )


if __name__ == "__main__":
    for version, source in CANDIDATES:
        build(version, source)
