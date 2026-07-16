from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Core reusable cleaning function
# ---------------------------------------------------------------------------

def clean_text_column(
    series: pd.Series,
    strip: bool = True,
    lowercase: bool = True,
    remove_special: bool = False,
    mapping: dict | None = None,
) -> pd.Series:
    """Reusable text cleaning function — apply any combination of transforms.

    Designed to be called on any string column.  Each transformation is
    optional so the same function works for names (strip + lowercase),
    categories (strip + lowercase + mapping), and cities (strip + remove_special).

    Transformations are applied in a deliberate order:
      1. strip         — remove invisible whitespace before anything else
      2. lowercase     — normalise casing after whitespace is gone
      3. remove_special— regex after casing so pattern is predictable
      4. mapping       — standardise spelling variants on the normalised value

    Args:
        series: A pandas Series of string / object dtype.
        strip: Remove leading and trailing whitespace.
               " Electronics " → "Electronics"
        lowercase: Convert all characters to lowercase.
                   "JOHN", "John", "john" → all become "john"
        remove_special: Strip everything except letters, numbers, and spaces
                        using regex [^a-zA-Z0-9 ].
                        "São Paulo" → "So Paulo", "Montréal" → "Montreal"
        mapping: Dict mapping raw (already normalised) values to canonical form.
                 {"b2b": "B2B", "b 2 b": "B2B"} → all variants → "B2B"

    Returns:
        Cleaned Series with the same index as the input.
    """
    result = series.copy()

    # 1. Strip whitespace — invisible but breaks exact matching and groupby
    if strip:
        result = result.str.strip()

    # 2. Normalize casing — "JOHN" and "john" are the same entity
    if lowercase:
        result = result.str.lower()

    # 3. Remove special characters — [^a-zA-Z0-9 ] means NOT letters/digits/space
    if remove_special:
        result = result.str.replace('[^a-zA-Z0-9 ]', '', regex=True)

    # 4. Map spelling variations to a canonical form
    if mapping:
        result = result.map(mapping)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean_string_columns(
    df: pd.DataFrame,
    column_config: dict[str, dict],
) -> tuple[pd.DataFrame, dict]:
    """Apply string cleaning transforms to multiple columns and log every change.

    Each entry in column_config maps a column name to a dict of cleaning
    options understood by clean_text_column():
        {
            "region": {"strip": True, "lowercase": True},
            "segment": {"strip": True, "lowercase": True,
                        "mapping": {"b2b": "B2B", "b 2 b": "B2B"}},
            "city":    {"strip": True, "remove_special": True},
        }

    Skips columns not present in the DataFrame (with a logged warning) so
    the function does not crash when the config is reused across datasets
    with different schemas.

    Args:
        df: Input DataFrame (typically post-deduplication or post-type-enforcement).
        column_config: Mapping of column name → cleaning options dict.

    Returns:
        (cleaned DataFrame, cleaning audit report dict)
    """
    working_df = df.copy()
    log: list[dict] = []

    print("\n── String Cleaning & Text Normalisation ───────────────────────")

    for col, options in column_config.items():
        if col not in working_df.columns:
            log.append({"column": col, "status": "skipped",
                        "reason": f"Column '{col}' not found in DataFrame."})
            print(f"  ⚠  [{col}] not found — skipped")
            continue

        # Only clean object/string columns — skip numeric/datetime by design
        if not pd.api.types.is_object_dtype(working_df[col]):
            log.append({"column": col, "status": "skipped",
                        "reason": f"Column dtype is {working_df[col].dtype}, not object/string."})
            print(f"  ⚠  [{col}] dtype {working_df[col].dtype} — skipped (not a string column)")
            continue

        before_sample = working_df[col].dropna().unique()[:5].tolist()

        working_df[col] = clean_text_column(
            working_df[col],
            strip=options.get("strip", True),
            lowercase=options.get("lowercase", True),
            remove_special=options.get("remove_special", False),
            mapping=options.get("mapping", None),
        )

        after_sample = working_df[col].dropna().unique()[:5].tolist()

        # Count how many values actually changed
        changed_count = int((df[col].astype(str) != working_df[col].astype(str)).sum())

        entry = {
            "column": col,
            "status": "cleaned",
            "transformations": {
                "strip": options.get("strip", True),
                "lowercase": options.get("lowercase", True),
                "remove_special": options.get("remove_special", False),
                "mapping_applied": options.get("mapping") is not None,
            },
            "values_changed": changed_count,
            "sample_before": before_sample,
            "sample_after": after_sample,
        }
        log.append(entry)

        transforms = []
        if options.get("strip", True):
            transforms.append("strip")
        if options.get("lowercase", True):
            transforms.append("lowercase")
        if options.get("remove_special", False):
            transforms.append("remove_special")
        if options.get("mapping"):
            transforms.append("mapping")

        print(f"  ✓ [{col}] {' → '.join(transforms)}  ({changed_count} value(s) changed)")

    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "columns_cleaned": sum(1 for e in log if e["status"] == "cleaned"),
        "columns_skipped": sum(1 for e in log if e["status"] == "skipped"),
        "details": log,
    }

    return working_df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_string_cleaning_log(report: dict, output_path: str | Path) -> None:
    """Persist the string cleaning audit report to JSON.

    Args:
        report: Report dict returned by clean_string_columns().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ String cleaning report saved → {path}")
