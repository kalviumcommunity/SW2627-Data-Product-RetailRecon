from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# String → Datetime
# ---------------------------------------------------------------------------

def convert_date_columns(
    df: pd.DataFrame,
    date_columns: list[str],
    date_format: str = "%Y-%m-%d",
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert string date columns to datetime using an explicit format string.

    Never relies on pandas auto-inference.  The string "01-02-2025" is
    ambiguous without a format: MM-DD (US) or DD-MM (European).  Letting
    pandas guess silently corrupts data on some machines while working fine
    on others.  Always pass format explicitly.

    Args:
        df: Input DataFrame (typically post-imputation).
        date_columns: Names of columns to convert.
        date_format: strftime format string matching the source data,
                     e.g. '%Y-%m-%d' for '2025-01-15'.

    Returns:
        (converted DataFrame, list of per-column conversion log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for col in date_columns:
        if col not in working_df.columns:
            log.append({"column": col, "conversion": "datetime",
                        "status": "skipped",
                        "reason": f"Column '{col}' not found."})
            continue

        before_dtype = str(working_df[col].dtype)

        # ALWAYS specify format — never rely on inference
        working_df[col] = pd.to_datetime(working_df[col], format=date_format)

        after_dtype = str(working_df[col].dtype)
        log.append({
            "column": col,
            "conversion": "datetime",
            "status": "success",
            "before_dtype": before_dtype,
            "after_dtype": after_dtype,
            "format_used": date_format,
        })
        print(f"  ✓ [{col}] {before_dtype} → {after_dtype}  (format: '{date_format}')")

    return working_df, log


# ---------------------------------------------------------------------------
# Currency String → Float
# ---------------------------------------------------------------------------

def convert_currency_columns(
    df: pd.DataFrame,
    currency_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Strip currency symbols / commas then convert to numeric float.

    Cannot sum currency text.  Must be numeric for all revenue calculations.
    Pattern '[$,]' removes dollar signs and thousands-separators.

    Args:
        df: Input DataFrame.
        currency_columns: Names of columns containing currency strings
                          like '$1,250.00'.

    Returns:
        (converted DataFrame, list of per-column conversion log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for col in currency_columns:
        if col not in working_df.columns:
            log.append({"column": col, "conversion": "float",
                        "status": "skipped",
                        "reason": f"Column '{col}' not found."})
            continue

        before_dtype = str(working_df[col].dtype)

        # Only apply string operations when the column is stored as object/text
        if pd.api.types.is_object_dtype(working_df[col]):
            # Strip symbols first, then convert
            working_df[col] = working_df[col].str.replace('[$,]', '', regex=True)

        working_df[col] = pd.to_numeric(working_df[col])

        after_dtype = str(working_df[col].dtype)
        log.append({
            "column": col,
            "conversion": "float",
            "status": "success",
            "before_dtype": before_dtype,
            "after_dtype": after_dtype,
        })
        print(f"  ✓ [{col}] {before_dtype} → {after_dtype}")

    return working_df, log


# ---------------------------------------------------------------------------
# Integer 0/1 → Boolean
# ---------------------------------------------------------------------------

def convert_boolean_columns(
    df: pd.DataFrame,
    boolean_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert integer 0/1 columns to proper Python boolean dtype.

    Models expect actual bool, not integer 0/1.  astype(bool) maps
    0 → False and any non-zero value → True consistently.

    Args:
        df: Input DataFrame.
        boolean_columns: Names of columns that represent boolean flags.

    Returns:
        (converted DataFrame, list of per-column conversion log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for col in boolean_columns:
        if col not in working_df.columns:
            log.append({"column": col, "conversion": "boolean",
                        "status": "skipped",
                        "reason": f"Column '{col}' not found."})
            continue

        before_dtype = str(working_df[col].dtype)

        working_df[col] = working_df[col].astype(bool)  # 0→False, 1→True

        after_dtype = str(working_df[col].dtype)
        log.append({
            "column": col,
            "conversion": "boolean",
            "status": "success",
            "before_dtype": before_dtype,
            "after_dtype": after_dtype,
        })
        print(f"  ✓ [{col}] {before_dtype} → {after_dtype}")

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
) -> tuple[pd.DataFrame, dict]:
    """Enforce correct data types across all configured columns.

    Runs conversions in order: dates → currency → booleans.
    Prints a before/after dtype comparison table so the impact of each
    conversion is visible in the pipeline log.

    Args:
        df: DataFrame after imputation.
        date_columns: Columns to convert to datetime.
        date_format: strftime format string for date columns.
        currency_columns: Columns containing currency strings to float.
        boolean_columns: Columns of integer 0/1 flags to bool.

    Returns:
        (type-enforced DataFrame, enforcement audit report dict)
    """
    date_columns = date_columns or []
    currency_columns = currency_columns or []
    boolean_columns = boolean_columns or []

    # Capture dtypes before any conversion
    before_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    all_logs: list[dict] = []

    print("\n── Type Enforcement ──────────────────────────────────────────")

    # 1. String → Datetime
    if date_columns:
        print(f"\n[1/3] Date columns → datetime  {date_columns}")
        df, date_log = convert_date_columns(df, date_columns, date_format=date_format)
        all_logs.extend(date_log)
    else:
        print("\n[1/3] Date columns: none — skipped")

    # 2. Currency → Float
    if currency_columns:
        print(f"\n[2/3] Currency columns → float  {currency_columns}")
        df, currency_log = convert_currency_columns(df, currency_columns)
        all_logs.extend(currency_log)
    else:
        print("\n[2/3] Currency columns: none — skipped")

    # 3. Integer → Boolean
    if boolean_columns:
        print(f"\n[3/3] Boolean columns → bool  {boolean_columns}")
        df, bool_log = convert_boolean_columns(df, boolean_columns)
        all_logs.extend(bool_log)
    else:
        print("\n[3/3] Boolean columns: none — skipped")

    # Before / after dtype comparison to validate all conversions succeeded
    after_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    changed = 0

    print("\n── Before / After dtype comparison ──────────────────────────")
    print(f"  {'Column':<28} {'Before':<20} {'After'}")
    print(f"  {'-'*28} {'-'*20} {'-'*20}")
    for col in before_dtypes:
        before = before_dtypes[col]
        after = after_dtypes.get(col, before)
        marker = "→" if before != after else "="
        if before != after:
            changed += 1
        print(f"  {col:<28} {before:<20} {marker}  {after}")

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
    """Persist the type enforcement audit report to a JSON file.

    Args:
        report: Report dict returned by enforce_types().
        output_path: Destination path for the JSON report.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Type enforcement report saved → {path}")
