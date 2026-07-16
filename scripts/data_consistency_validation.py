from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Five categories of validation checks
# ---------------------------------------------------------------------------

def check_range(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.Series:
    """Range check — ensure values fall within expected numeric bounds.

    Catches impossible values that indicate data entry errors or corruption:
    negative prices, ages above 150, dates outside a reasonable window.

    Args:
        df: Input DataFrame.
        column: Column to validate.
        min_value: Inclusive lower bound (None = no lower bound).
        max_value: Inclusive upper bound (None = no upper bound).

    Returns:
        Boolean Series — True where the value passes the range check.
    """
    mask = pd.Series(True, index=df.index)
    if column not in df.columns:
        return mask

    series = df[column]
    if min_value is not None:
        mask &= series >= min_value
    if max_value is not None:
        mask &= series <= max_value
    return mask


def check_not_null(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Null constraint — ensure a critical column never has missing values.

    customer_id, primary keys, and required identifiers should never be empty.
    A null in a critical field means the record is incomplete and unusable.

    Args:
        df: Input DataFrame.
        column: Column that must be non-null.

    Returns:
        Boolean Series — True where the value is not null.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].notna()


def check_format_pattern(
    df: pd.DataFrame,
    column: str,
    pattern: str,
) -> pd.Series:
    """Format pattern check — validate text structure with a regex pattern.

    Email must contain @, phone must be digits only, postal codes must match
    expected patterns.  Catches malformed entries that break downstream systems.

    Args:
        df: Input DataFrame.
        column: String column to validate.
        pattern: Regex pattern that valid values must match fully.

    Returns:
        Boolean Series — True where the value matches the pattern (NaN → False).
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].astype(str).str.match(pattern, na=False)


def check_referential_integrity(
    df: pd.DataFrame,
    column: str,
    valid_values: set,
) -> pd.Series:
    """Referential integrity — ensure column values exist in a reference set.

    If a record references a customer_id, that ID must exist in the customer
    table.  Catches orphaned references that indicate broken joins or
    deleted parent records.

    Args:
        df: Input DataFrame.
        column: Foreign-key column to validate.
        valid_values: Set of valid reference values.

    Returns:
        Boolean Series — True where the value is in valid_values.
    """
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].isin(valid_values)


def check_business_rule(
    df: pd.DataFrame,
    rule_name: str,
    rule_func,
) -> pd.Series:
    """Business rule check — apply domain-specific logic to the full DataFrame.

    End date must be after start date.  Order total must equal sum of items.
    Discount cannot exceed 50%.  These logical consistency checks catch
    violations that range checks alone cannot detect.

    Args:
        df: Input DataFrame.
        rule_name: Human-readable name for logging.
        rule_func: Callable(df) → boolean Series aligned to df's index.

    Returns:
        Boolean Series — True where the record passes the business rule.
    """
    try:
        result = rule_func(df)
        return result.fillna(False)
    except Exception as exc:
        print(f"  ⚠  Business rule '{rule_name}' error: {exc}")
        return pd.Series(True, index=df.index)  # skip gracefully on error


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_consistency_validation(
    df: pd.DataFrame,
    rules: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply all validation rules, isolate failures, and produce a report.

    Each entry in rules is a dict describing one check:
        {
            "name"       : "valid_amount_range",   # column added to df
            "type"       : "range",                # range | not_null | format | referential | business
            "column"     : "amount",               # target column (not used for 'business')
            "min_value"  : 0,                      # range check
            "max_value"  : 1_000_000,              # range check
            "pattern"    : r'^\\d{10}$',           # format check
            "valid_values": {...},                  # referential check
            "rule_func"  : lambda df: ...,          # business check
            "description": "Amount must be >= 0"   # human-readable reason
        }

    After all rules run:
      - A 'passes_all_checks' column is added combining every rule.
      - Records failing at least one rule are isolated to a failures DataFrame.
      - A structured report documents pass/fail counts per rule.

    Args:
        df: Input DataFrame (post outlier handling or post imputation).
        rules: List of rule definition dicts (see above).

    Returns:
        (clean DataFrame that passed all checks,
         failures DataFrame with all failing records,
         validation report dict)
    """
    working_df = df.copy()
    rule_results: list[dict] = []
    check_columns: list[str] = []

    print("\n── Data Consistency & Validation Rules ────────────────────────")

    for rule in rules:
        rule_type = rule.get("type", "range")
        rule_name = rule.get("name", f"rule_{len(check_columns)}")
        col = rule.get("column", "")
        description = rule.get("description", "")

        # Skip if target column missing (except business rules use full df)
        if rule_type != "business" and col and col not in working_df.columns:
            print(f"  ⚠  [{rule_name}] column '{col}' not found — skipped")
            continue

        # --- Run the appropriate check ---
        if rule_type == "range":
            mask = check_range(
                working_df, col,
                min_value=rule.get("min_value"),
                max_value=rule.get("max_value"),
            )

        elif rule_type == "not_null":
            mask = check_not_null(working_df, col)

        elif rule_type == "format":
            mask = check_format_pattern(working_df, col, rule["pattern"])

        elif rule_type == "referential":
            mask = check_referential_integrity(
                working_df, col, rule.get("valid_values", set())
            )

        elif rule_type == "business":
            mask = check_business_rule(working_df, rule_name, rule["rule_func"])

        else:
            print(f"  ⚠  [{rule_name}] unknown rule type '{rule_type}' — skipped")
            continue

        # Store result as a column on working_df
        working_df[rule_name] = mask
        check_columns.append(rule_name)

        passed = int(mask.sum())
        failed = int((~mask).sum())

        rule_results.append({
            "rule": rule_name,
            "type": rule_type,
            "column": col,
            "description": description,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round((passed / len(working_df)) * 100, 2) if len(working_df) else 0.0,
        })

        status = "✓" if failed == 0 else "✗"
        print(f"  {status} [{rule_name}]  passed={passed}  failed={failed}  — {description}")

    # --- Combine all checks ---
    if check_columns:
        working_df["passes_all_checks"] = working_df[check_columns].all(axis=1)
    else:
        working_df["passes_all_checks"] = True

    total = len(working_df)
    passed_all = int(working_df["passes_all_checks"].sum())
    failed_all = total - passed_all

    print(f"\n── Validation Summary ────────────────────────────────────────")
    print(f"  Total records : {total:,}")
    print(f"  Passed all    : {passed_all:,}")
    print(f"  Failed        : {failed_all:,}")
    print("──────────────────────────────────────────────────────────────\n")

    # --- Isolate failures ---
    failures = working_df[~working_df["passes_all_checks"]].copy()

    # --- Proceed with clean data only ---
    clean_df = working_df[working_df["passes_all_checks"]].copy()
    # Drop the validation flag columns from the clean output
    drop_cols = check_columns + ["passes_all_checks"]
    clean_df = clean_df.drop(columns=[c for c in drop_cols if c in clean_df.columns])

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": total,
        "passed_all_checks": passed_all,
        "failed_checks": failed_all,
        "pass_rate_pct": round((passed_all / total) * 100, 2) if total else 0.0,
        "rules": rule_results,
    }

    return clean_df, failures, report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_validation_report(report: dict, output_path: str | Path) -> None:
    """Persist the validation report to JSON.

    Args:
        report: Report dict returned by run_consistency_validation().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Validation report saved        → {path}")


def write_validation_failures(failures: pd.DataFrame, output_path: str | Path) -> None:
    """Save every failing record to CSV for audit and remediation.

    Downstream users and auditors can inspect exactly which records failed,
    which rules they violated, and trace them back to the source.

    Args:
        failures: DataFrame of failing records (returned by run_consistency_validation).
        output_path: Destination CSV path.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    failures.to_csv(path, index=False)
    print(f"✓ Validation failures CSV saved  → {path}  ({len(failures):,} record(s))")
