from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Join type reference
# ---------------------------------------------------------------------------
# inner : keep only rows where key matches BOTH sides
#         → result is smaller than both inputs
#         → use when you want only complete records
#
# left  : keep ALL from left, matched from right
#         → result row count == left table (unless multiplicity)
#         → use when left is your source of truth, right is enrichment
#
# right : keep ALL from right, matched from left
#         → mirrors left; result row count == right table (unless multiplicity)
#         → use when right is the authoritative table
#
# outer : keep ALL from both sides, NaN where no match
#         → result >= max(left, right) rows
#         → use when you need complete coverage and can handle nulls
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unmatched key detection
# ---------------------------------------------------------------------------

def find_unmatched_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify keys present in one table but missing from the other.

    Unmatched keys reveal orphaned records, referential gaps, and
    data collection problems.  Left-only keys = records with no enrichment.
    Right-only keys = enrichment rows with no matching source record.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        key: Column name(s) used as the join key.

    Returns:
        (left_only, right_only) — rows whose keys are absent from the other side.
    """
    keys = [key] if isinstance(key, str) else list(key)

    left_keys = left[keys].drop_duplicates()
    right_keys = right[keys].drop_duplicates()

    # Left-only: keys in left that do not appear in right
    left_only = left_keys.merge(right_keys, on=keys, how="left", indicator=True)
    left_only = left[
        left[keys].apply(tuple, axis=1).isin(
            left_only.loc[left_only["_merge"] == "left_only", keys]
            .apply(tuple, axis=1)
        )
    ].copy()

    # Right-only: keys in right that do not appear in left
    right_only = right_keys.merge(left_keys, on=keys, how="left", indicator=True)
    right_only = right[
        right[keys].apply(tuple, axis=1).isin(
            right_only.loc[right_only["_merge"] == "left_only", keys]
            .apply(tuple, axis=1)
        )
    ].copy()

    return left_only, right_only


# ---------------------------------------------------------------------------
# Core validated merge
# ---------------------------------------------------------------------------

def validated_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str | list[str],
    how: str = "left",
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[pd.DataFrame, dict]:
    """Merge two DataFrames with explicit join type and full row-count validation.

    Steps:
      1. Print row counts BEFORE merge so changes are visible.
      2. Perform pd.merge with the specified explicit join type.
      3. Print row count AFTER and compute the delta.
      4. Detect unmatched keys on both sides.
      5. Explain what the row count change means for this join type.

    Row count arithmetic tells the story:
      - Result < both inputs       → inner join filtered records
      - Result ≈ left row count    → left join (expected behavior)
      - Result > max(left, right)  → multiplicity from shared keys or outer join

    Args:
        left: Left (primary) DataFrame.
        right: Right (enrichment) DataFrame.
        key: Column name(s) to join on.
        how: Join type — 'inner', 'left', 'right', or 'outer'.
        left_name: Label for the left table in the audit report.
        right_name: Label for the right table in the audit report.

    Returns:
        (merged DataFrame, join validation report dict)
    """
    keys = [key] if isinstance(key, str) else list(key)
    rows_left = len(left)
    rows_right = len(right)

    print(f"\n── Merge: {left_name} ⟕ {right_name}  (how='{how}', key={keys}) ──")
    print(f"  {left_name:<20}: {rows_left:>8,} rows")
    print(f"  {right_name:<20}: {rows_right:>8,} rows")

    # Perform merge — join type MUST be explicit, never rely on default
    merged = pd.merge(left, right, on=keys, how=how)
    rows_merged = len(merged)
    row_delta = rows_merged - rows_left

    print(f"  {'merged result':<20}: {rows_merged:>8,} rows  "
          f"({'+'if row_delta >= 0 else ''}{row_delta:,} vs left)")

    # Explain what the row count change means
    if how == "inner":
        explanation = (
            f"Inner join: only {rows_merged:,} rows where key matched both sides. "
            f"{rows_left - rows_merged:,} left row(s) had no match in {right_name}."
        )
    elif how == "left":
        explanation = (
            f"Left join: all {rows_left:,} left rows retained. "
            f"Row count {'increased' if row_delta > 0 else 'unchanged'} by {abs(row_delta):,} "
            f"due to {'multiplicity (one left key matched multiple right rows)' if row_delta > 0 else 'no multiplicity'}."
        )
    elif how == "right":
        explanation = (
            f"Right join: all {rows_right:,} right rows retained. "
            f"Delta of {row_delta:,} vs left table."
        )
    else:  # outer
        explanation = (
            f"Outer join: all rows from both sides kept. "
            f"Result ({rows_merged:,}) >= max({rows_left:,}, {rows_right:,}). "
            f"NaN filled where no match."
        )

    print(f"  ℹ  {explanation}")

    # Detect unmatched keys
    left_unmatched, right_unmatched = find_unmatched_keys(left, right, keys)
    print(f"  Unmatched {left_name:<15}: {len(left_unmatched):,} key(s)")
    print(f"  Unmatched {right_name:<15}: {len(right_unmatched):,} key(s)")

    report = {
        "left_table": left_name,
        "right_table": right_name,
        "join_key": keys,
        "join_type": how,
        "rows_left": rows_left,
        "rows_right": rows_right,
        "rows_merged": rows_merged,
        "row_delta_vs_left": row_delta,
        "unmatched_left_keys": len(left_unmatched),
        "unmatched_right_keys": len(right_unmatched),
        "explanation": explanation,
    }

    return merged, report


# ---------------------------------------------------------------------------
# Multi-source orchestrator
# ---------------------------------------------------------------------------

def run_multi_source_merge(
    sources: list[dict],
) -> tuple[pd.DataFrame, dict]:
    """Merge multiple DataFrames sequentially with validation at each step.

    Each entry in sources defines one merge operation:
        {
            "df"        : pd.DataFrame,        # table to merge in
            "key"       : "customer_id",       # join key(s)
            "how"       : "left",              # join type
            "name"      : "orders",            # label for reporting
            "reason"    : "Enrich customers with order history"
        }

    The first entry is the base (left) table.  Each subsequent entry is
    merged into the running result, building up the unified view step by step.

    Args:
        sources: List of source dicts as described above.

    Returns:
        (unified merged DataFrame, full audit report dict)
    """
    if not sources:
        raise ValueError("At least one source is required.")

    print("\n══ Multi-Source Merge Pipeline ══════════════════════════════")

    base = sources[0]
    result_df = base["df"].copy()
    base_name = base.get("name", "base")
    merge_reports: list[dict] = []

    print(f"\n  Base table: '{base_name}'  ({len(result_df):,} rows)")

    for source in sources[1:]:
        right_df = source["df"]
        key = source["key"]
        how = source.get("how", "left")
        name = source.get("name", "unknown")
        reason = source.get("reason", "No reason provided.")

        print(f"\n  Merging '{name}'  reason: {reason}")

        result_df, step_report = validated_merge(
            left=result_df,
            right=right_df,
            key=key,
            how=how,
            left_name=base_name,
            right_name=name,
        )
        step_report["reason"] = reason
        merge_reports.append(step_report)

    print(f"\n══ Final unified table: {len(result_df):,} rows, "
          f"{len(result_df.columns)} columns ═══════════════════\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "base_table": base_name,
        "merge_steps": len(merge_reports),
        "final_rows": len(result_df),
        "final_columns": len(result_df.columns),
        "steps": merge_reports,
    }

    return result_df, report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_merge_report(report: dict, output_path: str | Path) -> None:
    """Persist the merge validation report to JSON.

    Args:
        report: Report dict returned by run_multi_source_merge().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Merge validation report saved  → {path}")


def write_unmatched_keys(
    left_unmatched: pd.DataFrame,
    right_unmatched: pd.DataFrame,
    output_dir: str | Path,
    left_name: str = "left",
    right_name: str = "right",
) -> None:
    """Save unmatched records from both sides to CSV for investigation.

    Unmatched key files answer: "Why does my merge lose records?" or
    "Why are there orphaned rows?"  They are essential for root-cause
    analysis and remediation decisions.

    Args:
        left_unmatched: Rows from left with no matching key in right.
        right_unmatched: Rows from right with no matching key in left.
        output_dir: Directory to write files into.
        left_name: Label used in the output filename.
        right_name: Label used in the output filename.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    left_path = out / f"unmatched_{left_name}.csv"
    right_path = out / f"unmatched_{right_name}.csv"

    left_unmatched.to_csv(left_path, index=False)
    right_unmatched.to_csv(right_path, index=False)

    print(f"✓ Unmatched {left_name} keys saved  → {left_path}  ({len(left_unmatched):,} row(s))")
    print(f"✓ Unmatched {right_name} keys saved  → {right_path}  ({len(right_unmatched):,} row(s))")
