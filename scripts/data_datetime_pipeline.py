from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1: Parse string timestamps → datetime
# ---------------------------------------------------------------------------

def parse_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: list[str],
    date_format: str = "%Y-%m-%d",
) -> tuple[pd.DataFrame, list[dict]]:
    """Convert string timestamp columns to datetime using an explicit format.

    Parse first — always.  Without datetime type the .dt accessor does not
    exist, .resample() raises an error, and all temporal arithmetic fails.
    Explicit format prevents silent data corruption from ambiguous strings
    (e.g. "01-02-2025" is Jan 2 in the US and Feb 1 in Europe).

    Args:
        df: Input DataFrame.
        datetime_columns: Column names that contain date/timestamp strings.
        date_format: strftime format matching the source data,
                     e.g. '%Y-%m-%d' or '%Y-%m-%d %H:%M:%S'.

    Returns:
        (parsed DataFrame, list of per-column parse log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for col in datetime_columns:
        if col not in working_df.columns:
            log.append({"column": col, "status": "skipped",
                        "reason": f"Column '{col}' not found."})
            continue

        before_dtype = str(working_df[col].dtype)

        # Parse first - always specify format, never rely on inference
        working_df[col] = pd.to_datetime(working_df[col], format=date_format)

        after_dtype = str(working_df[col].dtype)
        log.append({
            "column": col,
            "status": "parsed",
            "before_dtype": before_dtype,
            "after_dtype": after_dtype,
            "format_used": date_format,
        })
        print(f"  ✓ [{col}] {before_dtype} → {after_dtype}  (format: '{date_format}')")

    return working_df, log


# ---------------------------------------------------------------------------
# Step 2: Extract time-based features using the .dt accessor
# ---------------------------------------------------------------------------

def extract_datetime_features(
    df: pd.DataFrame,
    datetime_column: str,
    extract_day_of_week: bool = True,
    extract_hour: bool = True,
    extract_week_num: bool = True,
    extract_month: bool = True,
    extract_quarter: bool = True,
    compute_days_since: bool = True,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Extract temporal features from a parsed datetime column.

    Uses the .dt accessor — only available after the column has been parsed
    to datetime type.  Each feature is optional so the function works on
    date-only columns (no hour) and timestamp columns alike.

    Features added:
      {col}_day_of_week    — 'Monday', 'Tuesday', … (human-readable)
      {col}_dow_numeric    — 0=Monday … 6=Sunday
      {col}_hour           — 0–23 (only meaningful for timestamp columns)
      {col}_week_num       — ISO week number 1–53
      {col}_month          — 1–12
      {col}_quarter        — 1–4
      {col}_days_since     — integer days between reference_date and the event

    Args:
        df: DataFrame with a parsed datetime column.
        datetime_column: Name of the datetime column to extract features from.
        extract_day_of_week: Add day name and numeric day-of-week columns.
        extract_hour: Add hour-of-day column (0–23).
        extract_week_num: Add ISO week number column.
        extract_month: Add month number column.
        extract_quarter: Add fiscal quarter column.
        compute_days_since: Add days-since-event column using datetime arithmetic.
        reference_date: Reference point for days_since (defaults to today).

    Returns:
        (DataFrame with new feature columns, list of feature log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    if datetime_column not in working_df.columns:
        return working_df, [{"column": datetime_column, "status": "skipped",
                             "reason": "Column not found."}]

    if not pd.api.types.is_datetime64_any_dtype(working_df[datetime_column]):
        return working_df, [{"column": datetime_column, "status": "skipped",
                             "reason": f"Column dtype is {working_df[datetime_column].dtype},"
                                       f" not datetime. Call parse_datetime_columns first."}]

    col = datetime_column
    features_added: list[str] = []

    # Day of week — 'Monday', 'Tuesday', etc.
    if extract_day_of_week:
        working_df[f"{col}_day_of_week"] = working_df[col].dt.day_name()
        # Numeric version: 0=Monday, 6=Sunday
        working_df[f"{col}_dow_numeric"] = working_df[col].dt.dayofweek
        features_added += [f"{col}_day_of_week", f"{col}_dow_numeric"]

    # Hour of day — 0-23 representing midnight to 11 pm
    if extract_hour:
        working_df[f"{col}_hour"] = working_df[col].dt.hour
        features_added.append(f"{col}_hour")

    # ISO week number — 1 to 53
    if extract_week_num:
        working_df[f"{col}_week_num"] = working_df[col].dt.isocalendar().week.astype("int64")
        features_added.append(f"{col}_week_num")

    # Month number — 1 to 12
    if extract_month:
        working_df[f"{col}_month"] = working_df[col].dt.month
        features_added.append(f"{col}_month")

    # Quarter — 1 to 4
    if extract_quarter:
        working_df[f"{col}_quarter"] = working_df[col].dt.quarter
        features_added.append(f"{col}_quarter")

    # Days since event — datetime arithmetic produces timedelta, .dt.days extracts int
    if compute_days_since:
        ref = reference_date if reference_date is not None else pd.Timestamp.now()
        working_df[f"{col}_days_since"] = (ref - working_df[col]).dt.days
        features_added.append(f"{col}_days_since")

    log.append({
        "column": datetime_column,
        "status": "features_extracted",
        "features_added": features_added,
        "feature_count": len(features_added),
    })

    for feat in features_added:
        print(f"  ✓ [{feat}] extracted")

    return working_df, log


# ---------------------------------------------------------------------------
# Step 3: Time-series aggregation with resample
# ---------------------------------------------------------------------------

def build_time_series_aggregation(
    df: pd.DataFrame,
    datetime_column: str,
    value_column: str,
    frequency: str = "W",
    agg_func: str = "sum",
) -> pd.DataFrame:
    """Resample a numeric column by time frequency for trend analysis.

    .resample() requires a datetime index — this function sets the datetime
    column as the index, resamples, then returns a clean summary DataFrame.

    Common frequency aliases:
      'D' = calendar day   'W' = week     'ME' = month end
      'QE' = quarter end   'YE' = year end 'h' = hour

    Args:
        df: DataFrame with a parsed datetime column.
        datetime_column: Name of the datetime column to use as the time index.
        value_column: Numeric column to aggregate (e.g. 'amount').
        frequency: Resample frequency alias (default 'W' for weekly).
        agg_func: Aggregation function — 'sum', 'mean', 'count', 'min', 'max'.

    Returns:
        Resampled DataFrame with datetime index and aggregated values.
    """
    if datetime_column not in df.columns:
        raise ValueError(f"Datetime column '{datetime_column}' not found.")
    if value_column not in df.columns:
        raise ValueError(f"Value column '{value_column}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        raise ValueError(
            f"Column '{datetime_column}' must be datetime dtype. "
            f"Call parse_datetime_columns first."
        )

    # Set datetime as index — required for .resample() to work
    df_ts = df.set_index(datetime_column)

    resampled = getattr(df_ts[value_column].resample(frequency), agg_func)()

    print(f"  ✓ Resampled '{value_column}' by '{frequency}' using '{agg_func}' "
          f"→ {len(resampled)} period(s)")

    return resampled.reset_index()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_datetime_pipeline(
    df: pd.DataFrame,
    datetime_columns: list[str],
    date_format: str = "%Y-%m-%d",
    feature_column: str | None = None,
    resample_value_column: str | None = None,
    resample_frequency: str = "W",
    resample_agg: str = "sum",
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Run the full datetime pipeline: parse → extract features → resample.

    Args:
        df: Input DataFrame.
        datetime_columns: Columns to parse to datetime.
        date_format: strftime format string for parsing.
        feature_column: Which datetime column to extract features from.
                        Defaults to the first entry in datetime_columns.
        resample_value_column: Numeric column to aggregate in the resample step.
                               If None, the resample step is skipped.
        resample_frequency: Resample frequency alias ('W', 'D', 'ME', etc.).
        resample_agg: Aggregation function for resample ('sum', 'mean', etc.).

    Returns:
        (feature-enriched DataFrame, resampled summary DataFrame or None, report dict)
    """
    print("\n── Date & Time Transformation Pipeline ────────────────────────")

    # --- Step 1: Parse ---
    print(f"\n[1/3] Parsing datetime columns: {datetime_columns}")
    df, parse_log = parse_datetime_columns(df, datetime_columns, date_format=date_format)

    # --- Step 2: Extract features ---
    feat_col = feature_column or (datetime_columns[0] if datetime_columns else None)
    feature_log: list[dict] = []

    if feat_col:
        print(f"\n[2/3] Extracting time features from '{feat_col}'")
        df, feature_log = extract_datetime_features(df, feat_col)
    else:
        print("\n[2/3] Feature extraction: no column configured — skipped")

    # --- Step 3: Resample ---
    resampled_df: pd.DataFrame | None = None
    resample_log: dict = {}

    if resample_value_column and feat_col:
        print(f"\n[3/3] Resampling '{resample_value_column}' by '{resample_frequency}' ({resample_agg})")
        try:
            resampled_df = build_time_series_aggregation(
                df,
                datetime_column=feat_col,
                value_column=resample_value_column,
                frequency=resample_frequency,
                agg_func=resample_agg,
            )
            resample_log = {
                "status": "success",
                "datetime_column": feat_col,
                "value_column": resample_value_column,
                "frequency": resample_frequency,
                "agg_func": resample_agg,
                "periods": len(resampled_df),
            }
        except Exception as exc:
            print(f"  ⚠  Resample skipped: {exc}")
            resample_log = {"status": "skipped", "reason": str(exc)}
    else:
        print("\n[3/3] Resample: no value column configured — skipped")
        resample_log = {"status": "skipped", "reason": "resample_value_column not set"}

    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "datetime_columns_parsed": [e["column"] for e in parse_log if e.get("status") == "parsed"],
        "features_extracted": next(
            (e.get("features_added", []) for e in feature_log if e.get("status") == "features_extracted"),
            [],
        ),
        "resample": resample_log,
        "parse_log": parse_log,
        "feature_log": feature_log,
    }

    return df, resampled_df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_datetime_pipeline_log(
    report: dict,
    resampled_df: pd.DataFrame | None,
    output_dir: str | Path,
) -> None:
    """Persist the datetime pipeline report and resampled summary to output/.

    Args:
        report: Report dict returned by run_datetime_pipeline().
        resampled_df: Resampled DataFrame (or None if step was skipped).
        output_dir: Directory to write files into.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "datetime_pipeline_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"✓ Datetime pipeline report saved → {report_path}")

    if resampled_df is not None and not resampled_df.empty:
        resample_path = out / "resampled_summary.csv"
        resampled_df.to_csv(resample_path, index=False)
        print(f"✓ Resampled summary saved        → {resample_path}")
