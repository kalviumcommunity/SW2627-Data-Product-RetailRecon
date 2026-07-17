from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def enforce_datetime(
    df: pd.DataFrame,
    column: str,
    fmt: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert a string column to datetime using an explicit format string.

    Args:
        df:     DataFrame to modify (a copy is returned).
        column: Name of the column to convert.
        fmt:    strftime format string, e.g. '%Y-%m-%d' or '%d/%m/%Y'.
                Always required — never rely on pandas to infer the format,
                as ambiguous dates (e.g. '01-02-2025') silently corrupt data.

    Returns:
        (modified DataFrame, conversion log entry)

    Raises:
        KeyError:   Column does not exist in the DataFrame.
        ValueError: One or more values cannot be parsed with the given format.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    before_dtype = str(df[column].dtype)
    df = df.copy()

    # errors='raise' ensures bad values surface immediately rather than
    # silently becoming NaT and corrupting downstream date arithmetic.
    try:
        df[column] = pd.to_datetime(df[column], format=fmt, errors="raise")
    except Exception as exc:
        raise ValueError(
            f"enforce_datetime failed for column '{column}' with format '{fmt}': {exc}"
        ) from exc

    after_dtype = str(df[column].dtype)
    log: dict[str, Any] = {
        "column": column,
        "conversion": "string → datetime",
        "format_used": fmt,
        "before_dtype": before_dtype,
        "after_dtype": after_dtype,
        "status": "success",
    }
    return df, log


def enforce_currency(
    df: pd.DataFrame,
    column: str,
    strip_pattern: str = r"[$,£€¥\s]",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strip currency symbols and convert a column to float.

    Args:
        df:            DataFrame to modify (a copy is returned).
        column:        Name of the column to convert.
        strip_pattern: Regex pattern of characters to remove before casting.
                       Defaults to common currency symbols, commas, and spaces.

    Returns:
        (modified DataFrame, conversion log entry)

    Raises:
        KeyError: Column does not exist in the DataFrame.

    Notes:
        Uses errors='coerce' so unparseable residual values become NaN, which
        is visible and auditable rather than crashing the pipeline silently.
        The log records how many values were coerced so you can investigate.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    before_dtype = str(df[column].dtype)
    df = df.copy()

    cleaned = df[column].astype(str).str.replace(strip_pattern, "", regex=True).str.strip()
    converted = pd.to_numeric(cleaned, errors="coerce")

    coerced_count = int(converted.isnull().sum() - df[column].isnull().sum())
    coerced_count = max(coerced_count, 0)

    df[column] = converted
    after_dtype = str(df[column].dtype)

    log: dict[str, Any] = {
        "column": column,
        "conversion": "currency string → float",
        "strip_pattern": strip_pattern,
        "before_dtype": before_dtype,
        "after_dtype": after_dtype,
        "coerced_to_nan": coerced_count,
        "status": "success" if coerced_count == 0 else "success_with_coercion",
    }
    return df, log


def enforce_boolean(
    df: pd.DataFrame,
    column: str,
    mapping: dict[Any, bool] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert an integer or string column to proper Python bool.

    Args:
        df:      DataFrame to modify (a copy is returned).
        column:  Name of the column to convert.
        mapping: Optional explicit value → bool map.
                 Defaults to {0: False, 1: True, 'yes': True, 'no': False,
                              'true': True, 'false': False, 'y': True, 'n': False}.

    Returns:
        (modified DataFrame, conversion log entry)

    Raises:
        KeyError:   Column does not exist in the DataFrame.
        ValueError: Any value in the column is not covered by the mapping.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    default_mapping: dict[Any, bool] = {
        0: False, 1: True,
        "0": False, "1": True,
        "yes": True, "no": False,
        "true": True, "false": False,
        "y": True, "n": False,
    }
    active_mapping = mapping if mapping is not None else default_mapping

    before_dtype = str(df[column].dtype)
    df = df.copy()

    series = df[column].copy()
    if series.dtype == object:
        series = series.str.lower().str.strip()

    mapped = series.map(active_mapping)

    unmatched_mask = mapped.isnull() & series.notnull()
    if unmatched_mask.any():
        bad_vals = series[unmatched_mask].unique().tolist()
        raise ValueError(
            f"enforce_boolean failed for column '{column}': "
            f"unmapped value(s) {bad_vals}. "
            f"Extend the mapping or clean these values first."
        )

    df[column] = mapped.astype(bool)
    after_dtype = str(df[column].dtype)

    log: dict[str, Any] = {
        "column": column,
        "conversion": "integer/string → boolean",
        "mapping_keys": [str(k) for k in active_mapping.keys()],
        "before_dtype": before_dtype,
        "after_dtype": after_dtype,
        "status": "success",
    }
    return df, log


# ---------------------------------------------------------------------------
# Before / after dtype comparison
# ---------------------------------------------------------------------------

def compare_dtypes(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Return a column-by-column dtype diff between two DataFrames.

    Only columns where the dtype changed are included in the output.
    """
    changes: dict[str, dict[str, str]] = {}
    for col in before.columns:
        b = str(before[col].dtype)
        a = str(after[col].dtype)
        if b != a:
            changes[col] = {"before": b, "after": a}
    return changes


# ---------------------------------------------------------------------------
# Batch enforcement
# ---------------------------------------------------------------------------

def enforce_types(
    df: pd.DataFrame,
    type_map: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply multiple type conversions defined in a declarative type_map.

    Args:
        df:       Raw DataFrame.
        type_map: Dict mapping column name → conversion spec.
                  Each spec must have a 'type' key: 'datetime', 'currency', or 'boolean'.

    Returns:
        (converted DataFrame, list of per-column conversion logs)
    """
    logs: list[dict[str, Any]] = []
    result = df.copy()

    HANDLERS = {
        "datetime": _apply_datetime,
        "currency": _apply_currency,
        "boolean":  _apply_boolean,
    }

    for column, spec in type_map.items():
        conversion_type = spec.get("type", "").lower()
        if conversion_type not in HANDLERS:
            raise ValueError(
                f"Unknown type '{conversion_type}' for column '{column}'. "
                f"Supported: {list(HANDLERS.keys())}"
            )
        result, log = HANDLERS[conversion_type](result, column, spec)
        logs.append(log)

    return result, logs


def _apply_datetime(df: pd.DataFrame, column: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    fmt = spec.get("fmt")
    if not fmt:
        raise ValueError(
            f"'fmt' is required for datetime conversion of column '{column}'. "
            "Example: {'type': 'datetime', 'fmt': '%Y-%m-%d'}"
        )
    return enforce_datetime(df, column, fmt)


def _apply_currency(df: pd.DataFrame, column: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    kwargs: dict[str, Any] = {}
    if "strip_pattern" in spec:
        kwargs["strip_pattern"] = spec["strip_pattern"]
    return enforce_currency(df, column, **kwargs)


def _apply_boolean(df: pd.DataFrame, column: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    kwargs: dict[str, Any] = {}
    if "mapping" in spec:
        kwargs["mapping"] = spec["mapping"]
    return enforce_boolean(df, column, **kwargs)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def generate_type_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    conversion_logs: list[dict[str, Any]],
    report_path: str | Path = Path("../output/type_enforcement_report.json"),
) -> dict[str, Any]:
    """Write a structured JSON report summarising all type conversions."""
    dtype_diff = compare_dtypes(df_before, df_after)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "columns_converted": len(conversion_logs),
        "dtype_changes": dtype_diff,
        "conversion_logs": conversion_logs,
        "before_dtypes": {col: str(dt) for col, dt in df_before.dtypes.items()},
        "after_dtypes":  {col: str(dt) for col, dt in df_after.dtypes.items()},
    }

    out_path = report_path if isinstance(report_path, Path) else Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print_type_summary(report)
    print(f"\n✓ Type enforcement report saved to {out_path}")

    return report


def _print_type_summary(report: dict[str, Any]) -> None:
    """Print a human-readable console summary of all type conversions."""
    print("\nTYPE ENFORCEMENT SUMMARY")
    print(f"Columns converted: {report['columns_converted']}")

    changes = report["dtype_changes"]
    if changes:
        print("\n── dtype changes ──")
        for col, diff in changes.items():
            print(f"  {col}: {diff['before']} → {diff['after']}")
    else:
        print("  No dtype changes recorded.")

    logs = report["conversion_logs"]
    issues = [lg for lg in logs if lg.get("status", "").startswith("success_with")]
    if issues:
        print(f"\n── Conversions with warnings ({len(issues)}) ──")
        for lg in issues:
            coerced = lg.get("coerced_to_nan", 0)
            print(f"  ⚠ [{lg['column']}] {lg['conversion']}: {coerced} value(s) coerced to NaN")
    else:
        print("\n✓ All conversions completed cleanly")
