import csv
import math
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUTS = [
    ROOT / "kernel_v22" / "cloud_output" / "submission.csv",
    ROOT / "kernel_v23" / "cloud_output" / "submission.csv",
    ROOT / "kernel_v24" / "cloud_output" / "submission.csv",
]
OUTPUTS = [
    ROOT / "kernel_v25_ensemble" / "cloud_output" / "submission.csv",
    ROOT / "kernel_v26_ensemble" / "cloud_output" / "submission.csv",
    ROOT / "kernel_v27_ensemble" / "cloud_output" / "submission.csv",
]
EXPECTED_ROWS = 6_980_503
SENTINEL = object()


def main() -> None:
    for path in INPUTS:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in OUTPUTS:
        path.parent.mkdir(parents=True, exist_ok=True)

    with (
        INPUTS[0].open(newline="", encoding="utf-8") as f22,
        INPUTS[1].open(newline="", encoding="utf-8") as f23,
        INPUTS[2].open(newline="", encoding="utf-8") as f24,
    ):
        readers = [csv.reader(file) for file in (f22, f23, f24)]
        headers = [next(reader) for reader in readers]
        if not headers[0] == headers[1] == headers[2]:
            raise ValueError("V22, V23, and V24 submission headers differ")

        header = headers[0]
        indices = {name: index for index, name in enumerate(header)}
        required = ("submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow")
        missing = [name for name in required if name not in indices]
        if missing:
            raise ValueError(f"Missing submission columns: {missing}")

        with (
            OUTPUTS[0].open("w", newline="", encoding="utf-8") as f25,
            OUTPUTS[1].open("w", newline="", encoding="utf-8") as f26,
            OUTPUTS[2].open("w", newline="", encoding="utf-8") as f27,
        ):
            writers = [csv.writer(file) for file in (f25, f26, f27)]
            for writer in writers:
                writer.writerow(header)

            count = 0
            for rows in zip_longest(*readers, fillvalue=SENTINEL):
                if SENTINEL in rows:
                    raise ValueError(f"Input row counts differ after row {count}")
                row22, row23, row24 = rows
                count += 1
                if any(len(row) != len(header) for row in rows):
                    raise ValueError(f"Invalid column count at row {count}")
                if not (
                    row22[indices["submission_id"]]
                    == row23[indices["submission_id"]]
                    == row24[indices["submission_id"]]
                    and row22[indices["task"]]
                    == row23[indices["task"]]
                    == row24[indices["task"]]
                ):
                    raise ValueError(f"Input rows are misaligned at row {count}")

                base_rows = [row23.copy(), row22.copy(), row24.copy()]
                values = [
                    (row23, row24),
                    (row22, row24),
                    (row22, row23, row24),
                ]
                for output_row, source_rows in zip(base_rows, values):
                    for column in ("speed_kmh", "flow_vph"):
                        index = indices[column]
                        numbers = [float(row[index]) for row in source_rows]
                        mean = sum(numbers) / len(numbers)
                        if not math.isfinite(mean):
                            raise ValueError(f"Non-finite {column} at row {count}")
                        output_row[index] = format(mean, ".15g")
                    output_row[indices["queue_pred"]] = row24[indices["queue_pred"]]
                    output_row[indices["path_flow"]] = row24[indices["path_flow"]]
                for writer, output_row in zip(writers, base_rows):
                    writer.writerow(output_row)

    if count != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} data rows; wrote {count}")
    print(f"Built three aligned ensemble files with {count:,} data rows each.")


if __name__ == "__main__":
    main()
