"""Load and interpolate projectile drag coefficient tables."""

import csv
import math
import re
from pathlib import Path

import numpy as np

# Drag-table format constants.
DRG_SUFFIX = ".drg"
DRG_REQUIRED_HEADERS = ("NAME", "CAL", "BC", "REF")
G1_REFERENCE = "G1"
G1_MIN_CD_AT_MACH_ONE = 0.30
G1_MAX_CD_AT_MACH_ONE = 0.35
SELF_TEST_MACH_NUMBERS = (0.8, 1.0, 1.5, 2.0)


def _sort_and_validate(mach_values: list[float], cd_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate drag data and return arrays sorted by Mach.

    Args:
        mach_values: Mach-number values read from a table.
        cd_values: Drag-coefficient values read from a table.

    Returns:
        Mach and drag-coefficient arrays sorted by Mach.
    """
    if not mach_values:
        raise ValueError("Drag table contains no data rows")
    if len(mach_values) != len(cd_values):
        raise ValueError("Mach and Cd columns must have the same number of values")
    if not all(math.isfinite(value) for value in mach_values + cd_values):
        raise ValueError("Drag table values must be finite numbers")

    mach_array = np.asarray(mach_values, dtype=float)
    cd_array = np.asarray(cd_values, dtype=float)
    sort_indices = np.argsort(mach_array, kind="stable")
    return mach_array[sort_indices], cd_array[sort_indices]


def _parse_drg(path: str) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Parse a JBM DRG file and validate its metadata and data rows.

    Args:
        path: Path to the DRG file.

    Returns:
        Metadata dictionary and sorted Mach and Cd arrays.
    """
    metadata: dict[str, object] = {}
    mach_values: list[float] = []
    cd_values: list[float] = []

    with Path(path).open("r", encoding="utf-8-sig") as drg_file:
        for line_number, raw_line in enumerate(drg_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line_number == 1 and line.upper() != "# DRAG TABLE":
                    raise ValueError("DRG file must begin with '# DRAG TABLE'")
                if ":" in line:
                    header_name, header_value = line[1:].split(":", maxsplit=1)
                    header_name = header_name.strip().upper()
                    if header_name in DRG_REQUIRED_HEADERS:
                        metadata[header_name] = header_value.strip()
                continue

            fields = [field for field in re.split(r"[\s,]+", line) if field]
            if len(fields) != 2:
                raise ValueError(
                    f"Expected Mach and Cd on DRG line {line_number}, got: {line!r}"
                )
            try:
                mach_values.append(float(fields[0]))
                cd_values.append(float(fields[1]))
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric data on DRG line {line_number}: {line!r}"
                ) from error

    missing_headers = [header for header in DRG_REQUIRED_HEADERS if header not in metadata]
    if missing_headers:
        missing = ", ".join(missing_headers)
        raise ValueError(f"DRG file is missing required header(s): {missing}")

    try:
        metadata["bc"] = float(str(metadata["BC"]))
    except ValueError as error:
        raise ValueError("DRG BC header must contain a numeric value") from error

    parsed_metadata: dict[str, object] = {
        "name": metadata["NAME"],
        "caliber": metadata["CAL"],
        "bc": metadata["bc"],
        "reference": metadata["REF"],
    }
    mach_array, cd_array = _sort_and_validate(mach_values, cd_values)
    return parsed_metadata, mach_array, cd_array


def load_drag_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV or JBM DRG drag table and sort its points by Mach.

    Args:
        path: Path to a CSV file with mach and cd columns or a .drg file.

    Returns:
        Mach and Cd NumPy arrays sorted by Mach.
    """
    if Path(path).suffix.lower() == DRG_SUFFIX:
        _, mach_array, cd_array = _parse_drg(path)
        return mach_array, cd_array

    with Path(path).open("r", newline="", encoding="utf-8-sig") as csv_file:
        content = (line for line in csv_file if not line.lstrip().startswith("#"))
        reader = csv.DictReader(content)
        if reader.fieldnames is None:
            raise ValueError("CSV drag table must have a header row")

        normalized_headers = {
            header.strip().lower(): header for header in reader.fieldnames if header
        }
        if "mach" not in normalized_headers or "cd" not in normalized_headers:
            raise ValueError("CSV drag table must contain 'mach' and 'cd' columns")

        mach_values = []
        cd_values = []
        for line_number, row in enumerate(reader, start=2):
            try:
                mach_values.append(float(row[normalized_headers["mach"]]))
                cd_values.append(float(row[normalized_headers["cd"]]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid Mach or Cd value on CSV row {line_number}") from error

    return _sort_and_validate(mach_values, cd_values)


def get_cd(mach_arr: np.ndarray, cd_arr: np.ndarray, mach: float) -> float:
    """Interpolate Cd at a Mach number with flat extrapolation at table bounds.

    Args:
        mach_arr: Mach-number values in ascending order.
        cd_arr: Corresponding drag-coefficient values.
        mach: Mach number at which to interpolate Cd.

    Returns:
        Interpolated drag coefficient.
    """
    if mach_arr.ndim != 1 or cd_arr.ndim != 1:
        raise ValueError("Mach and Cd arrays must be one-dimensional")
    if mach_arr.size == 0 or mach_arr.size != cd_arr.size:
        raise ValueError("Mach and Cd arrays must be non-empty and have equal length")
    if not np.all(np.diff(mach_arr) >= 0.0):
        raise ValueError("Mach array must be sorted in ascending order")
    return float(np.interp(mach, mach_arr, cd_arr))


def load_drg_file(path: str) -> dict[str, object]:
    """Load a JBM DRG file and return its metadata and drag data.

    Args:
        path: Path to the JBM DRG file.

    Returns:
        Dictionary containing name, caliber, bc, reference, mach, and cd.
    """
    metadata, mach_array, cd_array = _parse_drg(path)
    return {**metadata, "mach": mach_array, "cd": cd_array}


if __name__ == "__main__":
    mach_array, cd_array = load_drag_table("data/g1.csv")
    for test_mach in SELF_TEST_MACH_NUMBERS:
        print(f"Cd at Mach {test_mach:.1f}: {get_cd(mach_array, cd_array, test_mach):.6f}")

    cd_at_mach_one = get_cd(mach_array, cd_array, 1.0)
    status = (
        "PASS"
        if G1_MIN_CD_AT_MACH_ONE <= cd_at_mach_one <= G1_MAX_CD_AT_MACH_ONE
        else "FAIL"
    )
    print(
        f"{status}: Cd at Mach 1.0 is {cd_at_mach_one:.6f} "
        f"(expected {G1_MIN_CD_AT_MACH_ONE:.2f}–{G1_MAX_CD_AT_MACH_ONE:.2f})"
    )
