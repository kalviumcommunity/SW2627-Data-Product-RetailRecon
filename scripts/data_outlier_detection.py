from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def detect_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using Z-score: how many standard deviations from the mean.

    Z-score measures distance from the mean in units of standard deviation.
    Values with |Z| > threshold are flagged as outliers.

    Best for: normally distributed data, scaled features.
    Sensitive to: extreme outliers that shift the mean itself.

    Args:
        series: Numeric pandas Series to test.
        threshold: Z-score cutoff — values beyond this are outliers (default 3.0).

    Returns:
        Boolean Series — True where the value is an outlier.
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    # Re-index to match original series so NaNs stay NaN, not False
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[series.dropna().index] = z_scores > threshold
    return outlier_mask


def detect_iqr_outliers(
    series: pd.Series,
    factor: float = 1.5,
) -> tuple[pd.Series, float, float]:
    """Detect outliers using the Interquartile Range (IQR) method.

    IQR = Q3 - Q1.  Boundaries: [Q1 - factor*IQR,  Q3 + factor*IQR].
    Values outside these boundaries are outliers.

    Best for: skewed distributions, non-normal data.
    Resistant to: extreme values shifting the detection boundary.

    Args:
        series: Numeric pandas Series to test.
        factor: Multiplier for the IQR boundary (default 1.5, use 3.0 for
                "far outliers" only).

    Returns:
        Tuple of (boolean outlier mask, lower_bound, upper_bound).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return outlier_mask, float(lower_bound), float(upper_bound)


# ---------------------------------------------------------------------------
# Handling strategies
# ---------------------------------------------------------------------------

def cap_outliers(
    series: pd.Series,
    lower_bound: float,
    upper_bound: float,
) -> pd.Series:
    """Strategy 1 — Cap at boundaries using clip().

    Extreme values are replaced with the boundary value.  All rows are
    preserved.  The influence of outliers on statistics is bounded.

    Values below lower_bound  → lower_bound
    Values above upper_bound  → upper_bound

    Args:
        series: Numeric Series to cap.
        lower_bound: Floor value.
        upper_bound: Ceiling value.

    Returns:
        Capped Series with same index.
    """
    return series.clip(lower=lower_bound, upper=upper_bound)


def remove_outlier_rows(
    df: pd.DataFrame,
    outlier_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strategy 2 — Remove rows flagged as outliers.

    Entire row is dropped when the outlier is a data collection error or
    truly invalid record.  Use only when the row has no analytical value.
    Documents removed rows for audit.

    Args:
        df: Full DataFrame.
        outlier_mask: Boolean Series (True = outlier) aligned to df's index.

    Returns:
        (clean DataFrame with outlier rows removed, DataFrame of removed rows)
    """
    removed = df[outlier_mask].copy()
    clean = df[~outlier_mask].reset_index(drop=True)
    return clean, removed


def flag_outliers(
    df: pd.DataFrame,
    outlier_mask: pd.Series,
    column: str,
) -> pd.DataFrame:
    """Strategy 3 — Flag outliers with a binary indicator column.

    All data is preserved.  A new integer column `is_{column}_outlier` is
    added: 1 = outlier, 0 = normal.  Downstream analysis can filter or
    weight flagged rows separately without losing any records.

    Args:
        df: Full DataFrame.
        outlier_mask: Boolean Series (True = outlier) aligned to df's index.
        column: Original column name — used to name the flag column.

    Returns:
        DataFrame with a new `is_{column}_outlier` column appended.
    """
    working_df = df.copy()
    working_df[f"is_{column}_outlier"] = outlier_mask.astype(int)
    return working_df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_outlier_detection(
    df: pd.DataFrame,
    column_config: dict[str, dict],
) -> tuple[pd.DataFrame, dict]:
    """Detect and handle outliers across multiple columns with full audit trail.

    Each column entry in column_config accepts:
        method   : 'zscore' or 'iqr'  (detection method)
        action   : 'cap', 'remove', or 'flag'  (handling strategy)
        threshold: Z-score cutoff (zscore method, default 3.0)
        factor   : IQR multiplier  (iqr method, default 1.5)
        reason   : Business justification string for the audit log

    Example config:
        {
            "amount": {
                "method": "iqr",
                "action": "cap",
                "factor": 1.5,
                "reason": "Cap extreme transaction amounts at IQR boundary"
            },
            "salary": {
                "method": "zscore",
                "action": "flag",
                "threshold": 3.0,
                "reason": "Flag executive salaries for separate reporting"
            }
        }

    Args:
        df: Input DataFrame (post type-enforcement or post-imputation).
        column_config: Per-column detection and handling configuration.

    Returns:
        (processed DataFrame, cleaning audit report dict)
    """
    working_df = df.copy()
    cleaning_log: list[dict] = []
    rows_before = len(working_df)

    print("\n── Outlier Detection & Handling ───────────────────────────────")

    for col, cfg in column_config.items():
        if col not in working_df.columns:
            cleaning_log.append({
                "column": col, "status": "skipped",
                "reason": f"Column '{col}' not found in DataFrame.",
            })
            print(f"  ⚠  [{col}] not found — skipped")
            continue

        if not pd.api.types.is_numeric_dtype(working_df[col]):
            cleaning_log.append({
                "column": col, "status": "skipped",
                "reason": f"Column dtype '{working_df[col].dtype}' is not numeric.",
            })
            print(f"  ⚠  [{col}] not numeric — skipped")
            continue

        method = cfg.get("method", "iqr")
        action = cfg.get("action", "flag")
        reason = cfg.get("reason", "No reason provided.")

        # --- Detection ---
        lower_bound: float | None = None
        upper_bound: float | None = None

        if method == "zscore":
            threshold = cfg.get("threshold", 3.0)
            outlier_mask = detect_zscore_outliers(working_df[col], threshold=threshold)
            detection_params = {"threshold": threshold}
        else:  # iqr
            factor = cfg.get("factor", 1.5)
            outlier_mask, lower_bound, upper_bound = detect_iqr_outliers(
                working_df[col], factor=factor
            )
            detection_params = {
                "factor": factor,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
            }

        outlier_count = int(outlier_mask.sum())

        print(f"\n  [{col}]  method={method}  action={action}  outliers={outlier_count}")

        if outlier_count == 0:
            cleaning_log.append({
                "column": col,
                "method": method,
                "action": action,
                "outliers_detected": 0,
                "rows_affected": 0,
                "status": "no_outliers",
                "detection_params": detection_params,
                "reason": reason,
            })
            print(f"    ✓ No outliers detected")
            continue

        # --- Handling ---
        rows_affected = outlier_count

        if action == "cap":
            if lower_bound is None or upper_bound is None:
                # Z-score capping: derive bounds from mean ± threshold*std
                threshold = cfg.get("threshold", 3.0)
                mean = working_df[col].mean()
                std = working_df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std

            working_df[col] = cap_outliers(working_df[col], lower_bound, upper_bound)
            print(f"    ✓ Capped {outlier_count} value(s) to "
                  f"[{lower_bound:.2f}, {upper_bound:.2f}]")

        elif action == "remove":
            working_df, removed = remove_outlier_rows(working_df, outlier_mask)
            rows_affected = len(removed)
            print(f"    ✓ Removed {rows_affected} row(s) containing outliers")

        elif action == "flag":
            working_df = flag_outliers(working_df, outlier_mask, col)
            flag_col = f"is_{col}_outlier"
            print(f"    ✓ Flagged {outlier_count} outlier(s) in new column '{flag_col}'")

        cleaning_log.append({
            "column": col,
            "method": method,
            "action": action,
            "outliers_detected": outlier_count,
            "rows_affected": rows_affected,
            "status": "handled",
            "detection_params": detection_params,
            "reason": reason,
        })

    rows_after = len(working_df)
    rows_removed = rows_before - rows_after

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Rows before : {rows_before:,}")
    print(f"  Rows after  : {rows_after:,}")
    print(f"  Rows removed: {rows_removed:,}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "cleaning_log": cleaning_log,
    }

    return working_df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_outlier_report(report: dict, output_path: str | Path) -> None:
    """Persist the outlier detection audit report to JSON.

    The cleaning_log inside the report documents column name, detection method,
    action taken, count of affected rows, and business reasoning — making
    every outlier decision traceable and reproducible.

    Args:
        report: Report dict returned by run_outlier_detection().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Outlier detection report saved → {path}")
