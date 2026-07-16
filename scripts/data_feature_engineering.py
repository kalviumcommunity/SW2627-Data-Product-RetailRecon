from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Type 1: Ratio features
# ---------------------------------------------------------------------------

def add_ratio_features(
    df: pd.DataFrame,
    ratio_config: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Create derived ratio columns that normalise for time or volume.

    Raw counts mean nothing without context.  50 transactions over 5 years
    is very different from 50 in one month.  Dividing by time or count
    exposes the rate — the real signal — rather than the raw accumulation.

    Each entry in ratio_config:
        {
            "name"       : "spend_per_transaction",   # new column name
            "numerator"  : "total_spent",             # existing column
            "denominator": "total_transactions",      # existing column or expression
            "description": "Average spend per transaction"
        }

    To divide by a time-based expression (e.g. days / 30 for months),
    pass a callable as denominator:
        "denominator": lambda df: df["days_as_customer"] / 30

    Args:
        df: Input DataFrame.
        ratio_config: List of ratio definition dicts.

    Returns:
        (DataFrame with new ratio columns, list of log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for cfg in ratio_config:
        name = cfg["name"]
        numerator = cfg["numerator"]
        description = cfg.get("description", name)

        if numerator not in working_df.columns:
            log.append({"feature": name, "status": "skipped",
                        "reason": f"Numerator column '{numerator}' not found."})
            print(f"  ⚠  [{name}] skipped — '{numerator}' not found")
            continue

        # Denominator can be a column name or a callable(df) → Series
        denom_cfg = cfg["denominator"]
        if callable(denom_cfg):
            denominator_series = denom_cfg(working_df)
            denom_label = "<expression>"
        else:
            if denom_cfg not in working_df.columns:
                log.append({"feature": name, "status": "skipped",
                            "reason": f"Denominator column '{denom_cfg}' not found."})
                print(f"  ⚠  [{name}] skipped — '{denom_cfg}' not found")
                continue
            denominator_series = working_df[denom_cfg]
            denom_label = denom_cfg

        # Divide — replace 0-denominator with NaN to avoid inf
        working_df[name] = working_df[numerator] / denominator_series.replace(0, float("nan"))

        log.append({
            "feature": name,
            "type": "ratio",
            "status": "created",
            "numerator": numerator,
            "denominator": denom_label,
            "description": description,
            "null_count": int(working_df[name].isna().sum()),
        })
        print(f"  ✓ [{name}] = {numerator} / {denom_label}  — {description}")

    return working_df, log


# ---------------------------------------------------------------------------
# Type 2: Binned / tiered features
# ---------------------------------------------------------------------------

def add_binned_features(
    df: pd.DataFrame,
    bin_config: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Create categorical tier columns using pd.cut (equal-width) or pd.qcut (quantile).

    Bins transform continuous values into interpretable segments:
    'high', 'medium', 'low' engagement tells a business story that raw
    transaction counts do not.  Two strategies are supported:

      cut  — equal-width bins with fixed boundary values.
              Use when boundaries have business meaning (e.g. 0-2 = low activity).
      qcut — quantile-based bins dividing data into equal-frequency groups.
              Use when you want balanced segments (e.g. quartiles, quintiles).

    Each entry in bin_config:
        {
            "name"    : "engagement_tier",
            "column"  : "transactions_per_month",
            "strategy": "cut",           # "cut" or "qcut"
            "bins"    : [0, 2, 10, float("inf")],   # cut only
            "q"       : 4,               # qcut only (number of quantiles)
            "labels"  : ["low", "medium", "high"],
            "description": "Customer engagement tier by transaction rate"
        }

    Args:
        df: Input DataFrame (should already have ratio features if referenced).
        bin_config: List of binning definition dicts.

    Returns:
        (DataFrame with new categorical bin columns, list of log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for cfg in bin_config:
        name = cfg["name"]
        column = cfg["column"]
        strategy = cfg.get("strategy", "cut")
        labels = cfg.get("labels")
        description = cfg.get("description", name)

        if column not in working_df.columns:
            log.append({"feature": name, "status": "skipped",
                        "reason": f"Source column '{column}' not found."})
            print(f"  ⚠  [{name}] skipped — '{column}' not found")
            continue

        try:
            if strategy == "qcut":
                q = cfg["q"]
                # duplicates="drop" prevents error when quantile boundaries coincide
                working_df[name] = pd.qcut(
                    working_df[column],
                    q=q,
                    labels=labels,
                    duplicates="drop",
                )
                bin_detail = f"qcut q={q}"
            else:
                bins = cfg["bins"]
                working_df[name] = pd.cut(
                    working_df[column],
                    bins=bins,
                    labels=labels,
                )
                bin_detail = f"cut bins={bins}"

            dist = working_df[name].value_counts(dropna=False).to_dict()
            log.append({
                "feature": name,
                "type": "binned",
                "status": "created",
                "source_column": column,
                "strategy": bin_detail,
                "labels": [str(l) for l in labels] if labels else None,
                "distribution": {str(k): int(v) for k, v in dist.items()},
                "description": description,
            })
            print(f"  ✓ [{name}] {bin_detail} on '{column}'  — {description}")

        except Exception as exc:
            log.append({"feature": name, "status": "error", "reason": str(exc)})
            print(f"  ✗ [{name}] error: {exc}")

    return working_df, log


# ---------------------------------------------------------------------------
# Type 3: Composite scores
# ---------------------------------------------------------------------------

def add_composite_scores(
    df: pd.DataFrame,
    score_config: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Build composite scores by combining multiple quantile-ranked signals.

    A composite score blends several indicators into one interpretable metric.
    The classic example is RFM (Recency × Frequency × Monetary) — each
    component is ranked into quantile buckets (1–5), then summed so that
    a high score means a high-value customer across all three dimensions.

    Each entry in score_config:
        {
            "name"       : "rfm_score",
            "components" : [
                {
                    "source"   : "days_since_purchase",
                    "q"        : 5,
                    "labels"   : [5, 4, 3, 2, 1],   # reversed: lower recency = higher score
                    "temp_col" : "recency_score",
                },
                {
                    "source"   : "purchase_count",
                    "q"        : 5,
                    "labels"   : [1, 2, 3, 4, 5],
                    "temp_col" : "frequency_score",
                },
                {
                    "source"   : "total_spent",
                    "q"        : 5,
                    "labels"   : [1, 2, 3, 4, 5],
                    "temp_col" : "monetary_score",
                },
            ],
            "keep_components": False,   # drop temp columns after summing
            "description": "RFM composite score (3-15, higher = better)"
        }

    Args:
        df: Input DataFrame.
        score_config: List of composite score definition dicts.

    Returns:
        (DataFrame with composite score columns, list of log entries)
    """
    working_df = df.copy()
    log: list[dict] = []

    for cfg in score_config:
        name = cfg["name"]
        components = cfg.get("components", [])
        keep = cfg.get("keep_components", False)
        description = cfg.get("description", name)
        temp_cols: list[str] = []
        skipped = False

        for comp in components:
            source = comp["source"]
            q = comp["q"]
            labels = comp["labels"]
            temp_col = comp.get("temp_col", f"_tmp_{source}")

            if source not in working_df.columns:
                log.append({"feature": name, "status": "skipped",
                            "reason": f"Component source '{source}' not found."})
                print(f"  ⚠  [{name}] skipped — component '{source}' not found")
                skipped = True
                break

            working_df[temp_col] = pd.qcut(
                working_df[source],
                q=q,
                labels=labels,
                duplicates="drop",
            )
            temp_cols.append(temp_col)

        if skipped:
            continue

        # Sum all components — cast to int first so arithmetic works
        working_df[name] = sum(
            working_df[col].astype(float).astype("Int64") for col in temp_cols
        )

        if not keep:
            working_df = working_df.drop(columns=temp_cols)

        score_min = int(working_df[name].min())
        score_max = int(working_df[name].max())

        log.append({
            "feature": name,
            "type": "composite",
            "status": "created",
            "components": [c.get("temp_col", c["source"]) for c in components],
            "score_range": [score_min, score_max],
            "description": description,
        })
        print(f"  ✓ [{name}] composite of {len(components)} component(s)  "
              f"range [{score_min}, {score_max}]  — {description}")

    return working_df, log


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_feature_engineering(
    df: pd.DataFrame,
    ratio_config: list[dict] | None = None,
    bin_config: list[dict] | None = None,
    score_config: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Engineer all features in one pass: ratios → bins → composites.

    Order matters: bin and score configs can reference ratio columns, so
    ratios are always created first.

    Args:
        df: Clean, merged DataFrame ready for feature creation.
        ratio_config: List of ratio feature definitions.
        bin_config: List of binning / tiering definitions.
        score_config: List of composite score definitions.

    Returns:
        (feature-enriched DataFrame, engineering audit report dict)
    """
    ratio_config = ratio_config or []
    bin_config = bin_config or []
    score_config = score_config or []

    cols_before = set(df.columns)
    all_logs: list[dict] = []

    print("\n── Feature Engineering & Derived Business Columns ────────────")

    # 1. Ratio features
    if ratio_config:
        print(f"\n[1/3] Ratio features ({len(ratio_config)})")
        df, ratio_log = add_ratio_features(df, ratio_config)
        all_logs.extend(ratio_log)
    else:
        print("\n[1/3] Ratio features: none configured — skipped")

    # 2. Binned / tiered features
    if bin_config:
        print(f"\n[2/3] Binned / tiered features ({len(bin_config)})")
        df, bin_log = add_binned_features(df, bin_config)
        all_logs.extend(bin_log)
    else:
        print("\n[2/3] Binned features: none configured — skipped")

    # 3. Composite scores
    if score_config:
        print(f"\n[3/3] Composite scores ({len(score_config)})")
        df, score_log = add_composite_scores(df, score_config)
        all_logs.extend(score_log)
    else:
        print("\n[3/3] Composite scores: none configured — skipped")

    cols_after = set(df.columns)
    new_cols = sorted(cols_after - cols_before)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Features created : {len(new_cols)}")
    for col in new_cols:
        print(f"    + {col}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "features_created": len(new_cols),
        "new_columns": new_cols,
        "details": all_logs,
    }

    return df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_feature_engineering_log(report: dict, output_path: str | Path) -> None:
    """Persist the feature engineering audit report to JSON.

    Args:
        report: Report dict returned by run_feature_engineering().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Feature engineering report saved → {path}")
