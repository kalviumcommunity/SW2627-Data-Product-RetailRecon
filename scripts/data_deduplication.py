from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect_exact_duplicates(df: pd.DataFrame) -> dict[str, int | float]:
    """Count exact duplicates — rows where every field is identical.

    Exact duplicates are the simplest case: the same row was imported twice
    (e.g. a source system that sends data twice by mistake).  .duplicated()
    with default keep='first' marks every copy after the first as True.

    Args:
        df: Input DataFrame.

    Returns:
        Summary dict with count and percentage of exact duplicate rows.
    """
    exact_dup_count = int(df.duplicated().sum())
    return {
        "exact_duplicates": exact_dup_count,
        "exact_duplicate_pct": round((exact_dup_count / len(df)) * 100, 2) if len(df) else 0.0,
    }


def detect_near_duplicates(df: pd.DataFrame, key_columns: list[str]) -> dict[str, int | float]:
    """Count near-duplicates — rows sharing the same key column values.

    Near-duplicates share the same business key (e.g. customer_id + date)
    but differ in other fields.  They arise when the same transaction is
    recorded twice from different sources with slightly different amounts
    or descriptions.

    Args:
        df: Input DataFrame.
        key_columns: Column(s) that together form the business key.

    Returns:
        Summary dict with count of near-duplicate rows.
    """
    valid_keys = [c for c in key_columns if c in df.columns]
    if not valid_keys:
        return {"near_duplicates_on_keys": 0, "key_columns_used": []}

    near_dup_count = int(df.duplicated(subset=valid_keys, keep=False).sum())
    return {
        "near_duplicates_on_keys": near_dup_count,
        "key_columns_used": valid_keys,
    }


# ---------------------------------------------------------------------------
# Deduplication strategies
# ---------------------------------------------------------------------------

def _keep_most_complete(group: pd.DataFrame) -> pd.Series:
    """Return the row from a group that has the fewest null values."""
    return group.loc[group.isnull().sum(axis=1).idxmin()]


def deduplicate_exact(
    df: pd.DataFrame,
    keep: str = "first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicates — every field identical.

    Strategy options:
      'first' — keep the original record (most common; first entry is reliable)
      'last'  — keep the most recent copy (use when later rows are corrections)
      False   — drop all copies of any duplicated row

    Args:
        df: Input DataFrame.
        keep: Which copy to keep: 'first', 'last', or False.

    Returns:
        (deduplicated DataFrame, DataFrame of removed rows for audit)
    """
    mask = df.duplicated(keep=keep)
    removed = df[mask].copy()
    deduped = df.drop_duplicates(keep=keep).reset_index(drop=True)
    return deduped, removed


def deduplicate_near(
    df: pd.DataFrame,
    key_columns: list[str],
    keep: str = "first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove near-duplicates by matching on key columns.

    Same key = same business entity recorded more than once.  Strategy:
      'first'     — keep original entry
      'last'      — keep most recent update
      'complete'  — keep the row with fewest nulls (best for merging partial
                    records from multiple sources)

    Args:
        df: Input DataFrame.
        key_columns: Columns that form the business key, e.g. ['customer_id', 'date'].
        keep: 'first', 'last', or 'complete'.

    Returns:
        (deduplicated DataFrame, DataFrame of removed rows for audit)
    """
    valid_keys = [c for c in key_columns if c in df.columns]
    if not valid_keys:
        return df.copy(), pd.DataFrame(columns=df.columns)

    if keep == "complete":
        # Group by key, pick the most complete row, rebuild the DataFrame
        deduped = (
            df.groupby(valid_keys, sort=False, group_keys=False)
            .apply(_keep_most_complete)
            .reset_index(drop=True)
        )
    else:
        deduped = df.drop_duplicates(subset=valid_keys, keep=keep).reset_index(drop=True)

    removed_mask = ~df.index.isin(deduped.index) if keep != "complete" else (
        df.duplicated(subset=valid_keys, keep=False) &
        ~df.index.isin(deduped.index)
    )
    removed = df[~df.index.isin(deduped.index)].copy()
    return deduped, removed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_deduplication(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    exact_keep: str = "first",
    near_keep: str = "first",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Detect and remove both exact and near-duplicates with full audit trail.

    Runs in two passes:
      1. Exact deduplication — drop rows where every field is identical.
      2. Near deduplication  — drop rows sharing the same business key.

    Logs before/after row counts, removal percentages, and preserves every
    removed record so any question of "where did that record go?" can be
    answered with certainty.

    Args:
        df: DataFrame after type enforcement.
        key_columns: Columns forming the business key for near-duplicate detection,
                     e.g. ['customer_id', 'date'].  Defaults to ['customer_id'].
        exact_keep: Strategy for exact duplicates — 'first', 'last', or False.
        near_keep:  Strategy for near-duplicates  — 'first', 'last', or 'complete'.

    Returns:
        (deduplicated DataFrame, audit report dict, DataFrame of all removed rows)
    """
    key_columns = key_columns or ["customer_id"]
    rows_before = len(df)

    print("\n── Duplicate Detection & Deduplication ───────────────────────")

    # --- Detection summary ---
    exact_stats = detect_exact_duplicates(df)
    near_stats = detect_near_duplicates(df, key_columns)

    print(f"\n  Rows before         : {rows_before:,}")
    print(f"  Exact duplicates    : {exact_stats['exact_duplicates']:,}  "
          f"({exact_stats['exact_duplicate_pct']}%)")
    print(f"  Near-duplicates     : {near_stats['near_duplicates_on_keys']:,}  "
          f"(key columns: {near_stats['key_columns_used']})")

    all_removed: list[pd.DataFrame] = []

    # --- Pass 1: exact duplicates ---
    print(f"\n[1/2] Removing exact duplicates  (keep='{exact_keep}')")
    df, exact_removed = deduplicate_exact(df, keep=exact_keep)
    if not exact_removed.empty:
        exact_removed = exact_removed.copy()
        exact_removed["_duplicate_type"] = "exact"
        exact_removed["_removed_at"] = datetime.now().isoformat()
        all_removed.append(exact_removed)
    print(f"  ✓ {len(exact_removed):,} exact duplicate row(s) removed")

    # --- Pass 2: near-duplicates ---
    print(f"\n[2/2] Removing near-duplicates on {key_columns}  (keep='{near_keep}')")
    df, near_removed = deduplicate_near(df, key_columns=key_columns, keep=near_keep)
    if not near_removed.empty:
        near_removed = near_removed.copy()
        near_removed["_duplicate_type"] = "near"
        near_removed["_removed_at"] = datetime.now().isoformat()
        all_removed.append(near_removed)
    print(f"  ✓ {len(near_removed):,} near-duplicate row(s) removed")

    # --- Before / after comparison ---
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    removal_pct = round((rows_removed / rows_before) * 100, 2) if rows_before else 0.0

    print("\n── Before / After comparison ─────────────────────────────────")
    print(f"  Before  : {rows_before:,} rows")
    print(f"  After   : {rows_after:,} rows")
    print(f"  Removed : {rows_removed:,} ({removal_pct}%)")
    print("──────────────────────────────────────────────────────────────\n")

    # --- Audit report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_removed,
        "removal_pct": removal_pct,
        "exact_duplicates_found": exact_stats["exact_duplicates"],
        "near_duplicates_found": near_stats["near_duplicates_on_keys"],
        "key_columns_used": near_stats["key_columns_used"],
        "strategies": {
            "exact": exact_keep,
            "near": near_keep,
        },
    }

    removed_df = (
        pd.concat(all_removed, ignore_index=True)
        if all_removed
        else pd.DataFrame(columns=list(df.columns) + ["_duplicate_type", "_removed_at"])
    )

    return df, report, removed_df


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_deduplication_log(report: dict, output_path: str | Path) -> None:
    """Persist the deduplication audit report to JSON.

    Args:
        report: Report dict returned by run_deduplication().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Deduplication report saved  → {path}")


def write_removed_records(removed_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save every removed duplicate row to CSV for compliance and audit.

    Answers the question: "Where did record X go?" → "It was a duplicate,
    removed on date Y — here is the full record."

    Args:
        removed_df: DataFrame of removed rows (returned by run_deduplication).
        output_path: Destination CSV path.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    removed_df.to_csv(path, index=False)
    print(f"✓ Removed records audit CSV saved → {path}  ({len(removed_df):,} row(s))")
