from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------

def convert_date_columns(
    df: pd.DataFrame,
    date_columns: list[str],
    date_format: str = "%Y-%m-%d",
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert string date columns to datetime using an explicit format.

    Never relies on pandas auto-inference — an explicit format prevents silent
    data corruption when the same date string is ambiguous across locales
    (e.g. "01-02-2025" is Jan 2 in the US but Feb 1 in Europe).

    Args:
        df: Input DataFrame.
        date_columns: Column names that should be converted to datetime.
        date_format: strftime format string that matches the source data.

    Returns:
        Tuple of (converted DataFrame, list of conversion log entries).
    """
    working_df = df.copy()
    log: list[dict] = []

    for column in date_columns:
        if column not in working_df.columns:
            log.append(
                {
                    "column": column,
                    "conversion": "datetime",
                    "status": "skipped",
                    "reason": f"Column '{column}' not found in DataFrame.",
                }
            )
            continue

        before_dtype = str(working_df[column].dtype)

        try:
            working_df[column] = pd.to_datetime(
                working_df[column],
                format=date_format,
                errors="raise",  # raise immediately — fail fast, not silently
            )
            after_dtype = str(working_df[column].dtype)
            log.append(
                {
                    "column": column,
                    "conversion": "datetime",
                    "status": "success",
                    "before_dtype": before_dtype,
                    "after_dtype": after_dtype,
                    "format_used": date_format,
                }
            )
            print(f"✓ [{column}] {before_dtype} → {after_dtype}  (format: {date_format})")
        except Exception as exc:
            raise ValueError(
                f"Type enforcement failed for column '{column}': "
                f"could not parse value as datetime with format '{date_format}'. "
                f"Original error: {exc}"
            ) from exc

    return working_df, log


def convert_currency_columns(
    df: pd.DataFrame,
    currency_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Strip currency symbols / commas and convert to float.

    Uses errors='coerce' so invalid values become NaN (visible) rather than
    raising silently or crashing mid-pipeline.

    Args:
        df: Input DataFrame.
        currency_columns: Column names that contain currency strings (e.g. "$1,250.00").

    Returns:
        Tuple of (converted DataFrame, list of conversion log entries).
    """
    working_df = df.copy()
    log: list[dict] = []

    for column in currency_columns:
        if column not in working_df.columns:
            log.append(
                {
                    "column": column,
                    "conversion": "float",
                    "status": "skipped",
                    "reason": f"Column '{column}' not found in DataFrame.",
                }
            )
            continue

        before_dtype = str(working_df[column].dtype)

        # Only strip symbols when the column is stored as text; if it is already
        # numeric, skip the string replacement step to avoid a TypeError.
        if pd.api.types.is_object_dtype(working_df[column]):
            working_df[column] = (
                working_df[column]
                .astype(str)
                .str.replace(r"[$,£€¥]", "", regex=True)
                .str.strip()
            )

        coerced = pd.to_numeric(working_df[column], errors="coerce")
        nan_introduced = int(coerced.isna().sum() - working_df[column].isna().sum())
        working_df[column] = coerced

        after_dtype = str(working_df[column].dtype)
        status = "success" if nan_introduced == 0 else "success_with_coercion"

        log.append(
            {
                "column": column,
                "conversion": "float",
                "status": status,
                "before_dtype": before_dtype,
                "after_dtype": after_dtype,
                "values_coerced_to_nan": nan_introduced,
            }
        )

        coercion_note = f"  ({nan_introduced} value(s) coerced to NaN)" if nan_introduced else ""
        print(f"✓ [{column}] {before_dtype} → {after_dtype}{coercion_note}")

    return working_df, log


def convert_boolean_columns(
    df: pd.DataFrame,
    boolean_columns: list[str],
    true_values: list | None = None,
    false_values: list | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert 0/1 integers (or yes/no strings) to proper Python booleans.

    Models expect actual bool dtype, not integer 0/1.  Using an explicit
    mapping avoids surprises when the column contains unexpected values.

    Args:
        df: Input DataFrame.
        boolean_columns: Column names that should become boolean.
        true_values: Values to treat as True  (default: [1, "1", "yes", "true", "y"]).
        false_values: Values to treat as False (default: [0, "0", "no", "false", "n"]).

    Returns:
        Tuple of (converted DataFrame, list of conversion log entries).
    """
    if true_values is None:
        true_values = [1, "1", "yes", "true", "y", "Yes", "True", "YES", "TRUE"]
    if false_values is None:
        false_values = [0, "0", "no", "false", "n", "No", "False", "NO", "FALSE"]

    bool_map = {v: True for v in true_values}
    bool_map.update({v: False for v in false_values})

    working_df = df.copy()
    log: list[dict] = []

    for column in boolean_columns:
        if column not in working_df.columns:
            log.append(
                {
                    "column": column,
                    "conversion": "boolean",
                    "status": "skipped",
                    "reason": f"Column '{column}' not found in DataFrame.",
                }
            )
            continue

        before_dtype = str(working_df[column].dtype)
        mapped = working_df[column].map(bool_map)

        # Detect values that were not in the mapping (NaN after map means unmapped)
        unmapped_mask = mapped.isna() & working_df[column].notna()
        unmapped_count = int(unmapped_mask.sum())

        if unmapped_count > 0:
            unmapped_values = working_df.loc[unmapped_mask, column].unique().tolist()
            raise ValueError(
                f"Type enforcement failed for column '{column}': "
                f"{unmapped_count} value(s) could not be mapped to boolean. "
                f"Unmapped values: {unmapped_values}. "
                f"Add them to true_values or false_values."
            )

        working_df[column] = mapped.astype("boolean")
        after_dtype = str(working_df[column].dtype)

        log.append(
            {
                "column": column,
                "conversion": "boolean",
                "status": "success",
                "before_dtype": before_dtype,
                "after_dtype": after_dtype,
            }
        )
        print(f"✓ [{column}] {before_dtype} → {after_dtype}")

    return working_df, log


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def enforce_types(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
    date_format: str = "%Y-%m-%d",
    currency_columns: list[str] | None = None,
    boolean_columns: list[str] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run all type conversions and return an audit report.

    Conversions are applied in order: dates → currency → booleans.
    Any conversion failure raises immediately (fail fast) with a clear
    message identifying the exact column and value that caused the problem.

    Args:
        df: Raw DataFrame after imputation.
        date_columns: Columns to convert to datetime.
        date_format: strftime format string for date columns.
        currency_columns: Columns to strip and convert to float.
        boolean_columns: Columns to convert to boolean.
        true_values: Custom list of values that map to True.
        false_values: Custom list of values that map to False.

    Returns:
        Tuple of (type-enforced DataFrame, enforcement report dict).
    """
    date_columns = date_columns or []
    currency_columns = currency_columns or []
    boolean_columns = boolean_columns or []

    before_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    all_logs: list[dict] = []

    print("\n── Type Enforcement ──────────────────────────────────────────")

    # 1. Dates
    if date_columns:
        print(f"\n[1/3] Date columns: {date_columns}")
        df, date_log = convert_date_columns(df, date_columns, date_format=date_format)
        all_logs.extend(date_log)
    else:
        print("\n[1/3] Date columns: none configured — skipped")

    # 2. Currency
    if currency_columns:
        print(f"\n[2/3] Currency columns: {currency_columns}")
        df, currency_log = convert_currency_columns(df, currency_columns)
        all_logs.extend(currency_log)
    else:
        print("\n[2/3] Currency columns: none configured — skipped")

    # 3. Booleans
    if boolean_columns:
        print(f"\n[3/3] Boolean columns: {boolean_columns}")
        df, bool_log = convert_boolean_columns(
            df,
            boolean_columns,
            true_values=true_values,
            false_values=false_values,
        )
        all_logs.extend(bool_log)
    else:
        print("\n[3/3] Boolean columns: none configured — skipped")

    after_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Summary table
    print("\n── Before / After dtype comparison ──────────────────────────")
    changed = 0
    for col in before_dtypes:
        before = before_dtypes[col]
        after = after_dtypes.get(col, before)
        marker = "→" if before != after else "="
        if before != after:
            changed += 1
        print(f"  {col:<30} {before:<20} {marker}  {after}")
    print(f"\n  {changed} column(s) converted")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "columns_converted": changed,
        "before_dtypes": before_dtypes,
        "after_dtypes": after_dtypes,
        "conversions": all_logs,
    }

    return df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_type_enforcement_log(report: dict, output_path: str | Path) -> None:
    """Persist the type enforcement report to JSON.

    Args:
        report: Report dict returned by enforce_types().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, default=str)

    print(f"✓ Type enforcement log saved → {path}")
