from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# KPI definition schema
# ---------------------------------------------------------------------------

@dataclass
class KPIDefinition:
    """Formal definition of a single KPI.

    Attributes:
        name:            Human-readable KPI name (unique identifier).
        formula:         Plain-English formula describing what is computed.
        data_source_cols: Columns from the dataset this KPI depends on.
        target_min:      Lower bound of the acceptable target range.
        target_max:      Upper bound of the acceptable target range.
        owner:           Team or person accountable for this metric.
        update_frequency: How often the KPI should be refreshed (e.g. 'Daily').
        unit:            Display unit string, e.g. '$', '%', 'users'.
        notes:           Any caveats, seasonality notes, or context.
    """
    name: str
    formula: str
    data_source_cols: list[str]
    target_min: float
    target_max: float
    owner: str
    update_frequency: str
    unit: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in KPI computation functions
# ---------------------------------------------------------------------------

def calculate_mau(
    df: pd.DataFrame,
    date_col: str = "date",
    customer_col: str = "customer_id",
    days: int = 30,
) -> float:
    """Monthly Active Users: distinct customers with a transaction in the last N days.

    Formula: COUNT(DISTINCT customer_id) WHERE date >= TODAY() - N days

    Args:
        df:           Transactions DataFrame (one row per transaction).
        date_col:     Name of the datetime column.
        customer_col: Name of the customer identifier column.
        days:         Lookback window in days (default 30).

    Returns:
        Count of unique active customers.
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    active_df = df[df[date_col] >= cutoff]
    return float(active_df[customer_col].nunique())


def calculate_revenue_per_customer(
    df: pd.DataFrame,
    amount_col: str = "amount",
    customer_col: str = "customer_id",
) -> float:
    """Average Revenue per Customer: total revenue divided by unique customer count.

    Formula: SUM(amount) / COUNT(DISTINCT customer_id)

    Args:
        df:           Transactions DataFrame.
        amount_col:   Name of the revenue column.
        customer_col: Name of the customer identifier column.

    Returns:
        Average revenue per unique customer, or 0.0 if no customers.
    """
    unique_customers = df[customer_col].nunique()
    if unique_customers == 0:
        return 0.0
    return float(df[amount_col].sum() / unique_customers)


def calculate_average_order_value(
    df: pd.DataFrame,
    amount_col: str = "amount",
) -> float:
    """Average Order Value (AOV): mean transaction amount.

    Formula: SUM(amount) / COUNT(transactions)

    Args:
        df:         Transactions DataFrame.
        amount_col: Name of the revenue column.

    Returns:
        Mean transaction amount, or 0.0 if no rows.
    """
    if df.empty:
        return 0.0
    return float(df[amount_col].mean())


def calculate_transaction_frequency(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
) -> float:
    """Transaction Frequency: average number of transactions per customer.

    Formula: COUNT(transactions) / COUNT(DISTINCT customer_id)

    Args:
        df:           Transactions DataFrame.
        customer_col: Name of the customer identifier column.

    Returns:
        Average transactions per customer, or 0.0 if no customers.
    """
    unique_customers = df[customer_col].nunique()
    if unique_customers == 0:
        return 0.0
    return float(len(df) / unique_customers)


def calculate_revenue_growth_rate(
    df: pd.DataFrame,
    amount_col: str = "amount",
    date_col: str = "date",
    periods: int = 2,
    freq: str = "ME",
) -> float:
    """Period-over-Period Revenue Growth Rate.

    Resamples revenue by the given frequency, then computes the % change
    between the most recent complete period and the one before it.

    Formula: (revenue_current_period - revenue_prior_period) / revenue_prior_period * 100

    Args:
        df:         Transactions DataFrame with a datetime date column.
        amount_col: Name of the revenue column.
        date_col:   Name of the datetime column.
        periods:    Number of most-recent periods to consider (default 2).
        freq:       Resample frequency string ('ME' = month-end, 'W' = weekly).

    Returns:
        Growth rate as a percentage (e.g. 5.2 means +5.2%).
        Returns 0.0 if fewer than two periods exist.
    """
    if df.empty or date_col not in df.columns:
        return 0.0

    monthly = df.set_index(date_col)[amount_col].resample(freq).sum()
    if len(monthly) < 2:
        return 0.0

    recent = monthly.iloc[-1]
    prior  = monthly.iloc[-2]
    if prior == 0:
        return 0.0

    return round(float((recent - prior) / prior * 100), 2)


def calculate_high_value_customer_rate(
    df: pd.DataFrame,
    amount_col: str = "amount",
    customer_col: str = "customer_id",
    threshold_pct: float = 0.8,
) -> float:
    """High-Value Customer Rate: share of customers above the revenue percentile threshold.

    Formula: COUNT(customers WHERE total_spend >= Nth percentile) / COUNT(DISTINCT customers) * 100

    Args:
        df:              Transactions DataFrame.
        amount_col:      Name of the revenue column.
        customer_col:    Name of the customer identifier column.
        threshold_pct:   Percentile threshold (default 0.8 = top 20% spenders).

    Returns:
        Percentage of customers classified as high-value.
    """
    if df.empty:
        return 0.0

    customer_spend = df.groupby(customer_col)[amount_col].sum()
    threshold_val  = customer_spend.quantile(threshold_pct)
    high_value     = (customer_spend >= threshold_val).sum()
    return round(float(high_value / len(customer_spend) * 100), 2)


# ---------------------------------------------------------------------------
# KPI registry — single source of truth for all metric definitions
# ---------------------------------------------------------------------------

def build_kpi_registry() -> dict[str, KPIDefinition]:
    """Return the project's canonical KPI definitions.

    All metrics computed by this module are defined here with their formal
    formula, target range, data dependencies, owner, and update frequency.
    Update this registry whenever a definition changes — never change
    computation logic silently.

    Returns:
        Dict mapping KPI key → KPIDefinition.
    """
    return {
        "mau": KPIDefinition(
            name="Monthly Active Users (MAU)",
            formula="COUNT(DISTINCT customer_id) WHERE date >= TODAY() - 30 days",
            data_source_cols=["customer_id", "date"],
            target_min=500,
            target_max=10_000,
            owner="Product Lead",
            update_frequency="Daily",
            unit="users",
            notes="Indicator of product engagement. Seasonal dips expected in Q4.",
        ),
        "rpc": KPIDefinition(
            name="Revenue per Customer",
            formula="SUM(amount) / COUNT(DISTINCT customer_id)",
            data_source_cols=["amount", "customer_id"],
            target_min=50.0,
            target_max=500.0,
            owner="Finance",
            update_frequency="Monthly",
            unit="$",
            notes="Measures sustainable revenue per relationship. Excludes refunds.",
        ),
        "aov": KPIDefinition(
            name="Average Order Value (AOV)",
            formula="SUM(amount) / COUNT(transactions)",
            data_source_cols=["amount"],
            target_min=20.0,
            target_max=300.0,
            owner="Sales",
            update_frequency="Weekly",
            unit="$",
            notes="Low AOV may indicate discount overuse or cart abandonment.",
        ),
        "txn_freq": KPIDefinition(
            name="Transaction Frequency",
            formula="COUNT(transactions) / COUNT(DISTINCT customer_id)",
            data_source_cols=["customer_id"],
            target_min=1.5,
            target_max=20.0,
            owner="Product Lead",
            update_frequency="Monthly",
            unit="txns/customer",
            notes="Higher frequency signals stronger retention. Track by cohort.",
        ),
        "revenue_growth": KPIDefinition(
            name="Revenue Growth Rate (MoM)",
            formula="(revenue_this_month - revenue_last_month) / revenue_last_month * 100",
            data_source_cols=["amount", "date"],
            target_min=0.0,
            target_max=50.0,
            owner="Finance",
            update_frequency="Monthly",
            unit="%",
            notes="Month-over-month. Negative values flag revenue contraction.",
        ),
        "hv_customer_rate": KPIDefinition(
            name="High-Value Customer Rate",
            formula="COUNT(customers >= 80th percentile spend) / COUNT(DISTINCT customers) * 100",
            data_source_cols=["amount", "customer_id"],
            target_min=15.0,
            target_max=30.0,
            owner="Marketing",
            update_frequency="Monthly",
            unit="%",
            notes="Tracks concentration risk. If >30%, revenue is too dependent on few customers.",
        ),
    }


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_all_kpis(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    customer_col: str = "customer_id",
    mau_days: int = 30,
) -> dict[str, float]:
    """Compute all registered KPIs from a transactions DataFrame.

    Args:
        df:           Cleaned transactions DataFrame.
        date_col:     Name of the datetime column.
        amount_col:   Name of the revenue column.
        customer_col: Name of the customer identifier column.
        mau_days:     Lookback window for MAU calculation.

    Returns:
        Dict mapping KPI key → computed value.
    """
    return {
        "mau":            calculate_mau(df, date_col, customer_col, mau_days),
        "rpc":            calculate_revenue_per_customer(df, amount_col, customer_col),
        "aov":            calculate_average_order_value(df, amount_col),
        "txn_freq":       calculate_transaction_frequency(df, customer_col),
        "revenue_growth": calculate_revenue_growth_rate(df, amount_col, date_col),
        "hv_customer_rate": calculate_high_value_customer_rate(df, amount_col, customer_col),
    }


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def validate_kpis_against_targets(
    kpi_values: dict[str, float],
    registry: dict[str, KPIDefinition],
) -> list[dict[str, Any]]:
    """Check each KPI value against its defined target range.

    Args:
        kpi_values: Output of compute_all_kpis.
        registry:   KPI registry (output of build_kpi_registry).

    Returns:
        List of result dicts, one per KPI, sorted — out-of-range first.
        Each dict contains: key, name, value, target_min, target_max,
        unit, status ('ON_TARGET' or 'OUT_OF_RANGE'), deviation.
    """
    results: list[dict[str, Any]] = []

    for key, value in kpi_values.items():
        defn = registry.get(key)
        if defn is None:
            continue

        on_target = defn.target_min <= value <= defn.target_max
        status    = "ON_TARGET" if on_target else "OUT_OF_RANGE"

        # Deviation: how far outside the range (0 if within target)
        if value < defn.target_min:
            deviation = round(value - defn.target_min, 2)
        elif value > defn.target_max:
            deviation = round(value - defn.target_max, 2)
        else:
            deviation = 0.0

        results.append({
            "key":        key,
            "name":       defn.name,
            "value":      round(value, 2),
            "target_min": defn.target_min,
            "target_max": defn.target_max,
            "unit":       defn.unit,
            "status":     status,
            "deviation":  deviation,
            "owner":      defn.owner,
        })

    # Out-of-range KPIs first so they are immediately visible
    return sorted(results, key=lambda r: (r["status"] == "ON_TARGET", r["key"]))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_kpi_dashboard(
    validation_results: list[dict[str, Any]],
    output_path: str | Path | None = None,
    title: str = "KPI Dashboard",
) -> None:
    """Render a horizontal bar chart comparing each KPI value to its target range.

    Green bars are on-target; red bars are out-of-range.
    Target range is shown as a shaded band so the gap is visible at a glance.

    Args:
        validation_results: Output of validate_kpis_against_targets.
        output_path:        If provided, saves as PNG; otherwise displays.
        title:              Chart title.
    """
    if not validation_results:
        print("⚠ No KPI results to plot.")
        return

    n      = len(validation_results)
    labels = [f"{r['name']}\n({r['unit']})" for r in validation_results]
    values = [r["value"] for r in validation_results]
    colors = ["#10b981" if r["status"] == "ON_TARGET" else "#ef4444" for r in validation_results]

    fig, ax = plt.subplots(figsize=(12, max(5, n * 1.1)))

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.8, height=0.5)

    # Shade target range
    for i, r in enumerate(reversed(validation_results)):
        y_pos = i
        ax.barh(
            y_pos, r["target_max"] - r["target_min"],
            left=r["target_min"],
            height=0.85,
            color="#d1fae5", alpha=0.5, zorder=0,
        )
        ax.axvline(r["target_min"], ymin=(y_pos) / n, ymax=(y_pos + 1) / n,
                   color="#6ee7b7", linewidth=1, linestyle="--", zorder=1)
        ax.axvline(r["target_max"], ymin=(y_pos) / n, ymax=(y_pos + 1) / n,
                   color="#6ee7b7", linewidth=1, linestyle="--", zorder=1)

    # Annotate values
    for bar, r in zip(bars, reversed(validation_results)):
        status_icon = "✓" if r["status"] == "ON_TARGET" else "✗"
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{status_icon} {r['value']:,.2f} {r['unit']}",
            va="center", fontsize=8, fontweight="bold",
        )

    ax.set_xlabel("Value")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(values) * 1.35 if values else 100)
    plt.tight_layout()

    if output_path:
        out = output_path if isinstance(output_path, Path) else Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ KPI dashboard chart saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_kpi_report(
    kpi_values: dict[str, float],
    validation_results: list[dict[str, Any]],
    registry: dict[str, KPIDefinition],
    report_path: str | Path = Path("../output/kpi_report.json"),
) -> dict[str, Any]:
    """Write a structured JSON KPI report and print a console summary.

    The report includes:
    - Computed KPI values with target validation results
    - Full KPI definitions from the registry (formula, owner, frequency)
    - Timestamp for audit and trend tracking

    Args:
        kpi_values:         Output of compute_all_kpis.
        validation_results: Output of validate_kpis_against_targets.
        registry:           KPI registry (output of build_kpi_registry).
        report_path:        Where to save the JSON report.

    Returns:
        The report dict.
    """
    on_target_count  = sum(1 for r in validation_results if r["status"] == "ON_TARGET")
    out_of_range     = [r for r in validation_results if r["status"] == "OUT_OF_RANGE"]

    report: dict[str, Any] = {
        "timestamp":        datetime.now().isoformat(),
        "kpis_computed":    len(kpi_values),
        "on_target":        on_target_count,
        "out_of_range":     len(out_of_range),
        "results":          validation_results,
        "definitions":      {k: asdict(v) for k, v in registry.items()},
    }

    out_path = report_path if isinstance(report_path, Path) else Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print_kpi_summary(report)
    print(f"\n✓ KPI report saved to {out_path}")

    return report


def _print_kpi_summary(report: dict[str, Any]) -> None:
    """Print a human-readable console summary of KPI results."""
    print("\nKPI DASHBOARD SUMMARY")
    print(f"KPIs computed: {report['kpis_computed']}  |  "
          f"On target: {report['on_target']}  |  "
          f"Out of range: {report['out_of_range']}")

    print("\n── KPI results ──")
    for r in report["results"]:
        icon   = "✓" if r["status"] == "ON_TARGET" else "✗"
        target = f"target: {r['target_min']}-{r['target_max']} {r['unit']}"
        dev    = f"  deviation: {r['deviation']:+.2f}" if r["deviation"] != 0 else ""
        print(f"  {icon} {r['name']}: {r['value']:,.2f} {r['unit']}  ({target}){dev}")


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def run_kpi_analysis(
    df: pd.DataFrame,
    date_col: str = "date",
    amount_col: str = "amount",
    customer_col: str = "customer_id",
    mau_days: int = 30,
    output_dir: str | Path = Path("../output"),
    report_path: str | Path = Path("../output/kpi_report.json"),
    save_plots: bool = True,
) -> dict[str, Any]:
    """Run the complete KPI analysis pipeline in one call.

    Steps:
    1. Load the KPI registry (definitions, targets, owners).
    2. Compute all KPI values from the DataFrame.
    3. Validate each value against its target range.
    4. Generate and save the KPI dashboard chart.
    5. Write and return the JSON report.

    Args:
        df:           Cleaned transactions DataFrame.
        date_col:     Name of the datetime column.
        amount_col:   Name of the revenue column.
        customer_col: Name of the customer identifier column.
        mau_days:     Lookback window for MAU.
        output_dir:   Directory for chart PNGs.
        report_path:  Path for the JSON report.
        save_plots:   If True, saves chart to output_dir; else displays it.

    Returns:
        The complete KPI report dict.
    """
    out_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    # 1. Registry
    registry = build_kpi_registry()

    # 2. Compute
    kpi_values = compute_all_kpis(df, date_col, amount_col, customer_col, mau_days)

    # 3. Validate
    validation_results = validate_kpis_against_targets(kpi_values, registry)

    # 4. Visualise
    chart_path = (out_dir / "kpi_dashboard.png") if save_plots else None
    plot_kpi_dashboard(validation_results, output_path=chart_path)

    # 5. Report
    return generate_kpi_report(
        kpi_values=kpi_values,
        validation_results=validation_results,
        registry=registry,
        report_path=report_path,
    )
