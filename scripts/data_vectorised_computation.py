from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vectorised operations
# ---------------------------------------------------------------------------

def minmax_normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisation — scale values to [0, 1].

    Formula: (x - min) / (max - min)

    This is the NumPy vectorised version.  The equivalent Python loop
    would call the interpreter once per row; NumPy compiles to C and
    processes all rows in one parallel operation.

    Returns an array of zeros when all values are identical (range = 0)
    to avoid division by zero.

    Args:
        arr: 1-D NumPy float array.

    Returns:
        Normalised array with values in [0, 1].
    """
    arr_min = arr.min()
    arr_max = arr.max()
    value_range = arr_max - arr_min
    if value_range == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - arr_min) / value_range


def zscore_normalise(arr: np.ndarray) -> np.ndarray:
    """Z-score standardisation — centre on mean, scale by standard deviation.

    Formula: (x - mean) / std

    Result has mean ≈ 0 and std ≈ 1.  Values represent how many standard
    deviations each element is from the mean — directly comparable across
    columns with different units.

    Returns zeros when std = 0 (constant column) to avoid division by zero.

    Args:
        arr: 1-D NumPy float array.

    Returns:
        Standardised array.
    """
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Rank each value as a percentile (0–100) using NumPy argsort.

    Useful for relative comparisons when absolute scale does not matter:
    "this transaction is in the 95th percentile of all amounts".

    Args:
        arr: 1-D NumPy float array.

    Returns:
        Float array with percentile ranks in [0, 100].
    """
    order = arr.argsort().argsort()          # stable rank (0-based)
    return (order / (len(arr) - 1)) * 100    # scale to 0-100


def clip_to_bounds(
    arr: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """Vectorised clip — replace values outside [lower, upper] with the boundary.

    Equivalent to pd.Series.clip but operating directly on a NumPy array for
    maximum throughput.  NumPy clip is a single compiled C call regardless of
    array size.

    Args:
        arr: 1-D NumPy float array.
        lower: Floor value (None = no lower bound).
        upper: Ceiling value (None = no upper bound).

    Returns:
        Clipped array.
    """
    return np.clip(arr, a_min=lower, a_max=upper)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_operation(func, *args, **kwargs):
    """Run func(*args, **kwargs), return (result, elapsed_seconds)."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def benchmark_loop_vs_vectorised(
    arr: np.ndarray,
    label: str = "column",
) -> dict:
    """Time a Python loop min-max normalisation against the NumPy version.

    The loop version calls the Python interpreter once per element — overhead
    compounds at scale.  The NumPy version is a single compiled C call.
    This function makes the difference visible and measurable.

    Args:
        arr: NumPy array to benchmark on.
        label: Column name used in the printout.

    Returns:
        Dict with loop_time_s, vectorised_time_s, and speedup_x.
    """
    # ── Loop version (intentionally slow — this is the baseline) ──────────
    def _loop_minmax(a: np.ndarray) -> list:
        a_min = a.min()
        a_max = a.max()
        value_range = a_max - a_min
        result = []
        for val in a:
            result.append(0.0 if value_range == 0 else (val - a_min) / value_range)
        return result

    _, loop_time = _time_operation(_loop_minmax, arr)

    # ── NumPy vectorised version ───────────────────────────────────────────
    _, vec_time = _time_operation(minmax_normalise, arr)

    speedup = (loop_time / vec_time) if vec_time > 0 else float("inf")

    print(f"  [{label}]  n={len(arr):,}")
    print(f"    Loop      : {loop_time:.4f}s")
    print(f"    NumPy     : {vec_time:.6f}s")
    print(f"    Speedup   : {speedup:,.0f}x")

    return {
        "column": label,
        "n_rows": len(arr),
        "loop_time_s": round(loop_time, 6),
        "vectorised_time_s": round(vec_time, 6),
        "speedup_x": round(speedup, 1),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_vectorised_computation(
    df: pd.DataFrame,
    column_config: list[dict],
    run_benchmark: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply vectorised NumPy operations to numeric columns and log results.

    Each entry in column_config defines one or more operations for a column:
        {
            "column"     : "amount",
            "operations" : ["minmax", "zscore", "percentile"],  # any subset
            "clip_lower" : 0,       # optional — clip before normalising
            "clip_upper" : None,
        }

    Supported operations:
      "minmax"     → {column}_minmax      (0–1 scaled)
      "zscore"     → {column}_zscore      (mean=0, std=1)
      "percentile" → {column}_percentile  (0–100 rank)

    Workflow per column:
      1. Extract column as NumPy array via .values (fastest path)
      2. Optionally clip to remove extreme values before normalising
      3. Apply each requested vectorised operation
      4. Write result back to DataFrame as a new column
      5. Optionally benchmark loop vs vectorised on this column

    Args:
        df: Input DataFrame (post feature engineering or post imputation).
        column_config: List of operation definition dicts.
        run_benchmark: If True, benchmark loop vs vectorised for each column.

    Returns:
        (DataFrame with new normalised columns, computation report dict)
    """
    working_df = df.copy()
    cols_before = set(working_df.columns)
    all_logs: list[dict] = []
    benchmarks: list[dict] = []

    print("\n── NumPy Vectorised Computation ───────────────────────────────")

    for cfg in column_config:
        col = cfg["column"]
        operations = cfg.get("operations", ["minmax"])
        clip_lower = cfg.get("clip_lower", None)
        clip_upper = cfg.get("clip_upper", None)

        if col not in working_df.columns:
            all_logs.append({"column": col, "status": "skipped",
                             "reason": f"Column '{col}' not found."})
            print(f"  ⚠  [{col}] not found — skipped")
            continue

        if not pd.api.types.is_numeric_dtype(working_df[col]):
            all_logs.append({"column": col, "status": "skipped",
                             "reason": f"dtype {working_df[col].dtype} is not numeric."})
            print(f"  ⚠  [{col}] not numeric — skipped")
            continue

        print(f"\n  [{col}]  operations={operations}  rows={len(working_df):,}")

        # Step 1: Convert to NumPy array — fastest path, avoids Pandas overhead
        arr = working_df[col].values.astype(float)

        # Step 2: Optional vectorised clip before normalising
        if clip_lower is not None or clip_upper is not None:
            arr = clip_to_bounds(arr, lower=clip_lower, upper=clip_upper)
            print(f"    → clipped to [{clip_lower}, {clip_upper}]")

        # Step 3 & 4: Apply each operation and write back to DataFrame
        new_cols: list[str] = []

        for op in operations:
            if op == "minmax":
                # FAST: NumPy vectorised — all rows in one compiled C call
                result = minmax_normalise(arr)
                out_col = f"{col}_minmax"

            elif op == "zscore":
                result = zscore_normalise(arr)
                out_col = f"{col}_zscore"

            elif op == "percentile":
                result = percentile_rank(arr)
                out_col = f"{col}_percentile"

            else:
                print(f"    ⚠  unknown operation '{op}' — skipped")
                continue

            # Integrate result back into Pandas DataFrame
            working_df[out_col] = result
            new_cols.append(out_col)
            print(f"    ✓ {out_col}  min={result.min():.4f}  max={result.max():.4f}")

        # Step 5: Benchmark loop vs vectorised
        if run_benchmark:
            bm = benchmark_loop_vs_vectorised(arr, label=col)
            benchmarks.append(bm)

        all_logs.append({
            "column": col,
            "status": "processed",
            "operations": operations,
            "new_columns": new_cols,
            "rows": len(working_df),
        })

    cols_after = set(working_df.columns)
    new_col_names = sorted(cols_after - cols_before)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  New columns created : {len(new_col_names)}")
    for c in new_col_names:
        print(f"    + {c}")
    if benchmarks:
        avg_speedup = sum(b["speedup_x"] for b in benchmarks) / len(benchmarks)
        print(f"  Avg speedup (NumPy vs loop) : {avg_speedup:,.0f}x")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "new_columns": new_col_names,
        "operations_log": all_logs,
        "benchmarks": benchmarks,
    }

    return working_df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_vectorised_computation_log(report: dict, output_path: str | Path) -> None:
    """Persist the vectorised computation report to JSON.

    Args:
        report: Report dict returned by run_vectorised_computation().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Vectorised computation report saved → {path}")
