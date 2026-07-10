from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def analyze_missing_before(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Summarize missing values before treatment."""
    summary: dict[str, dict[str, float | int]] = {}

    for column in df.columns:
        null_count = int(df[column].isna().sum())
        if null_count == 0:
            continue

        summary[column] = {
            "nulls": null_count,
            "null_pct": round((null_count / len(df)) * 100, 2) if len(df) else 0.0,
        }

    return summary


def _is_time_series_column(column_name: str, series: pd.Series, time_series_columns: list[str] | None) -> bool:
    if time_series_columns and column_name in time_series_columns:
        return True

    name = column_name.lower()
    return any(keyword in name for keyword in ("date", "time", "timestamp")) or pd.api.types.is_datetime64_any_dtype(series)


def impute_missing_values(
    df: pd.DataFrame,
    critical_columns: list[str] | None = None,
    time_series_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply type-aware missing value handling and return an audit log."""
    critical_columns = critical_columns or ["customer_id"]
    working_df = df.copy()
    before_nulls = analyze_missing_before(working_df)
    decisions: list[dict[str, object]] = []

    for column in critical_columns:
        if column in working_df.columns:
            null_count = int(working_df[column].isna().sum())
            if null_count > 0:
                working_df = working_df.dropna(subset=[column])
                decisions.append(
                    {
                        "column": column,
                        "strategy": "drop_rows",
                        "reason": "Critical identifier values cannot be imputed.",
                        "rows_affected": null_count,
                    }
                )

    for column in working_df.columns:
        if column in critical_columns:
            continue

        null_count = int(working_df[column].isna().sum())
        if null_count == 0:
            continue

        series = working_df[column]

        if _is_time_series_column(column, series, time_series_columns):
            working_df[column] = working_df[column].ffill()
            decisions.append(
                {
                    "column": column,
                    "strategy": "forward_fill",
                    "reason": "Time-ordered data should carry forward the previous observed value.",
                    "rows_affected": null_count,
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            working_df[column] = series.fillna(fill_value)
            decisions.append(
                {
                    "column": column,
                    "strategy": "median",
                    "reason": "Median is resistant to outliers in numeric business metrics.",
                    "fill_value": None if pd.isna(fill_value) else float(fill_value),
                    "rows_affected": null_count,
                }
            )
            continue

        mode_values = series.mode(dropna=True)
        if not mode_values.empty:
            fill_value = mode_values.iloc[0]
            working_df[column] = series.fillna(fill_value)
            decisions.append(
                {
                    "column": column,
                    "strategy": "mode",
                    "reason": "Most common category preserves the observed distribution.",
                    "fill_value": fill_value,
                    "rows_affected": null_count,
                }
            )
            continue

        working_df = working_df.dropna(subset=[column])
        decisions.append(
            {
                "column": column,
                "strategy": "drop_rows",
                "reason": "No stable fill value was available for this column.",
                "rows_affected": null_count,
            }
        )

    after_nulls = analyze_missing_before(working_df)
    report = {
        "timestamp": datetime.now().isoformat(),
        "before_nulls": before_nulls,
        "after_nulls": after_nulls,
        "decisions": decisions,
        "rows_before": int(len(df)),
        "rows_after": int(len(working_df)),
        "rows_removed": int(len(df) - len(working_df)),
    }

    return working_df, report


def write_imputation_log(report: dict, output_path: str | Path) -> None:
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, default=str)