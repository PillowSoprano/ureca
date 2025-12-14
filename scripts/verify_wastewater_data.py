"""Sanity checks for wastewater influent files and steady-state vectors.

Run this script before training or resilience evaluation to confirm the
expected files exist and have compatible shapes.

Example:
    python scripts/verify_wastewater_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

INFLOW_FILES = [
    ("Inf_dry_2006.txt", "dry"),
    ("inf_rain.txt", "rain"),
    ("inf_strm.txt", "storm"),
]
STEADY_STATE_FILE = "ss_open.txt"
EXPECTED_INFLOW_COLS = 15
# Some datasets provide a 145-length steady-state vector (matching the
# default `Nx=145` in `waste_water_system.py`), while others include an
# extended 156-length variant. Treat either length as valid so the check
# remains helpful across both sources.
EXPECTED_STEADY_STATE = (145, 156)


def _describe_array(name: str, arr: np.ndarray) -> str:
    if arr.ndim == 1:
        shape_str = f"vector length {arr.shape[0]}"
    else:
        shape_str = f"matrix shape {arr.shape}"
    return f"{name}: {shape_str}, dtype={arr.dtype}"


def _check_inflow(path: Path) -> Tuple[bool, str]:
    try:
        data = np.loadtxt(path)
    except OSError as exc:
        return False, f"missing ({exc})"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"failed to load ({exc})"

    if data.ndim != 2:
        return False, f"expected 2-D array, got {data.ndim} dims"

    rows, cols = data.shape
    if cols == EXPECTED_INFLOW_COLS:
        orientation = "time x 15 features"
    elif rows == EXPECTED_INFLOW_COLS:
        orientation = "15 x time (transpose recommended)"
    else:
        orientation = f"unexpected layout; wanted 15 columns"

    return True, f"{orientation}; {_describe_array(path.name, data)}"


def _check_steady_state(path: Path) -> Tuple[bool, str]:
    try:
        data = np.loadtxt(path)
    except OSError as exc:
        return False, f"missing ({exc})"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"failed to load ({exc})"

    if data.ndim != 1:
        return False, f"expected 1-D vector, got shape {data.shape}"

    length = data.shape[0]
    ok_len = length in EXPECTED_STEADY_STATE
    length_msg = (
        f"length OK ({length} entries)"
        if ok_len
        else f"unexpected length {length}; expected one of {EXPECTED_STEADY_STATE}"
    )
    return ok_len, f"{length_msg}; {_describe_array(path.name, data)}"


def run_checks(base: Path) -> int:
    status = 0
    print(f"Checking wastewater assets under: {base.resolve()}")

    for filename, scenario in INFLOW_FILES:
        ok, detail = _check_inflow(base / filename)
        flag = "OK" if ok else "WARN"
        print(f"[{flag}] inflow ({scenario}): {detail}")
        status |= 0 if ok else 1

    ok, detail = _check_steady_state(base / STEADY_STATE_FILE)
    flag = "OK" if ok else "WARN"
    print(f"[{flag}] steady-state: {detail}")
    status |= 0 if ok else 1

    if status:
        print("Some checks failed; fix the warnings above before training.")
    else:
        print("All wastewater files look consistent.")
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify wastewater data files")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path.cwd(),
        help="Directory containing influent and steady-state files",
    )
    args = parser.parse_args(argv)
    exit(run_checks(args.base))


if __name__ == "__main__":
    main()
