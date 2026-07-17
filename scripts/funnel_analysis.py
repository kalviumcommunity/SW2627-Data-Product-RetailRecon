from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------

def build_funnel_stages(
    df: pd.DataFrame,
    stage_definitions: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Count users at each funnel stage from a DataFrame.

    Each stage definition supports two filter modes:

    1. Flag column — count rows where a binary column equals a target value::

           {"column": "signup_completed", "value": 1}

    2. Threshold — count rows where a numeric column meets a condition::

           {"column": "purchase_count", "op": ">=", "value": 1}

    Args:
        df:                DataFrame containing one row per user.
        stage_definitions: Ordered dict mapping stage label → filter spec.
                           Order determines funnel sequence.

    Returns:
        Ordered dict mapping stage label → user count at that stage.

    Example::

        stage_definitions = {
            "Sign Up":        {"column": "signup_completed",  "value": 1},
            "Email Verified": {"column": "email_verified",    "value": 1},
            "Payment Added":  {"column": "payment_added",     "value": 1},
            "First Purchase": {"column": "first_purchase",    "value": 1},
        }
    """
    OPS = {
        ">=": lambda s, v: s >= v,
        "<=": lambda s, v: s <= v,
        ">":  lambda s, v: s > v,
        "<":  lambda s, v: s < v,
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
    }

    stages: dict[str, int] = {}
    for label, spec in stage_definitions.items():
        col = spec["column"]
        if col not in df.columns:
            raise KeyError(f"Stage '{label}': column '{col}' not found in DataFrame.")

        op = spec.get("op", "==")
        val = spec["value"]

        if op not in OPS:
            raise ValueError(f"Stage '{label}': unsupported operator '{op}'. Use one of {list(OPS)}.")

        stages[label] = int(OPS[op](df[col], val).sum())

    return stages


# ---------------------------------------------------------------------------
# Drop-off computation
# ---------------------------------------------------------------------------

def compute_dropoff(stages: dict[str, int]) -> pd.DataFrame:
    """Compute step-by-step drop-off metrics between consecutive funnel stages.

    For each consecutive pair of stages calculates:
    - users_lost      : absolute number who did not proceed
    - drop_rate_%     : % of users at the FROM stage who dropped
    - completion_rate_%: % of users at the FROM stage who continued

    Args:
        stages: Ordered dict of stage label → user count (output of build_funnel_stages).

    Returns:
        DataFrame with one row per transition, sorted by drop_rate descending
        so the biggest leak is always first.
    """
    if len(stages) < 2:
        raise ValueError("Funnel must have at least two stages to compute drop-off.")

    stage_names = list(stages.keys())
    stage_counts = list(stages.values())

    rows: list[dict[str, Any]] = []
    for i in range(len(stage_counts) - 1):
        from_count = stage_counts[i]
        to_count   = stage_counts[i + 1]
        lost       = from_count - to_count

        drop_rate        = (lost / from_count * 100) if from_count > 0 else 0.0
        completion_rate  = (to_count / from_count * 100) if from_count > 0 else 0.0

        rows.append({
            "from_stage":        stage_names[i],
            "to_stage":          stage_names[i + 1],
            "users_at_from":     from_count,
            "users_at_to":       to_count,
            "users_lost":        lost,
            "drop_rate_%":       round(drop_rate, 1),
            "completion_rate_%": round(completion_rate, 1),
        })

    return pd.DataFrame(rows).sort_values("drop_rate_%", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Biggest leak & business impact
# ---------------------------------------------------------------------------

def identify_biggest_leak(dropoff_df: pd.DataFrame) -> dict[str, Any]:
    """Return the single transition with the highest drop-off rate.

    Args:
        dropoff_df: Output of compute_dropoff.

    Returns:
        Dict with the worst transition's metrics plus a plain-English recommendation.
    """
    if dropoff_df.empty:
        raise ValueError("drop-off DataFrame is empty — nothing to analyse.")

    worst = dropoff_df.iloc[0].to_dict()
    worst["recommendation"] = (
        f"Priority: fix friction between '{worst['from_stage']}' and '{worst['to_stage']}'. "
        f"{worst['drop_rate_%']}% of users drop here ({worst['users_lost']:,} lost). "
        f"Recovering half of these users would add {worst['users_lost'] // 2:,} to the next stage."
    )
    return worst


def calculate_business_impact(
    dropoff_df: pd.DataFrame,
    revenue_per_user: float,
) -> pd.DataFrame:
    """Estimate the revenue impact of recovering lost users at each funnel step.

    For every transition, calculates:
    - revenue_lost_total  : all lost users × revenue_per_user
    - revenue_if_50pct_fix: revenue recovered if half of dropped users are retained

    Args:
        dropoff_df:       Output of compute_dropoff.
        revenue_per_user: Average revenue generated by a user who completes the funnel.

    Returns:
        dropoff_df with two additional revenue columns appended.
    """
    result = dropoff_df.copy()
    result["revenue_lost_total"]   = (result["users_lost"] * revenue_per_user).round(2)
    result["revenue_if_50pct_fix"] = (result["revenue_lost_total"] * 0.5).round(2)
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_funnel(
    stages: dict[str, int],
    output_path: str | Path | None = None,
    title: str = "Funnel Analysis",
) -> None:
    """Render a bar chart of user counts at each funnel stage.

    Bars are coloured from blue (top of funnel) to red (bottom), making
    drop-off visually obvious without needing a legend.

    Args:
        stages:      Ordered dict of stage label → user count.
        output_path: If provided, saves the figure as a PNG instead of displaying it.
        title:       Chart title.
    """
    labels = list(stages.keys())
    counts = list(stages.values())
    n      = len(labels)

    # Gradient palette: blue → green → amber → red
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    colors  = [palette[min(i, len(palette) - 1)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate each bar with its absolute count and % of funnel start
    top = counts[0] if counts[0] > 0 else 1
    for bar, count in zip(bars, counts):
        pct = count / top * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.01,
            f"{count:,}\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Users")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, top * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if output_path:
        out = output_path if isinstance(output_path, Path) else Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ Funnel chart saved to {out}")
        plt.close(fig)
    else:
        plt.show()


def plot_dropoff_rates(
    dropoff_df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Drop-Off Rate by Funnel Transition",
) -> None:
    """Render a horizontal bar chart ranked by drop-off rate.

    The biggest leak is always at the top, making prioritisation instant.

    Args:
        dropoff_df:  Output of compute_dropoff (already sorted descending).
        output_path: If provided, saves as PNG instead of displaying.
        title:       Chart title.
    """
    labels = [f"{r['from_stage']} → {r['to_stage']}" for _, r in dropoff_df.iterrows()]
    rates  = dropoff_df["drop_rate_%"].tolist()

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.8)))
    bar_colors = ["#ef4444" if r >= 30 else "#f59e0b" if r >= 15 else "#10b981" for r in rates]
    bars = ax.barh(labels[::-1], rates[::-1], color=bar_colors[::-1], edgecolor="white")

    for bar, rate in zip(bars, rates[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{rate}%", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Drop-Off Rate (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(rates) * 1.2 if rates else 100)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        out = output_path if isinstance(output_path, Path) else Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ Drop-off rate chart saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_funnel_report(
    stages: dict[str, int],
    dropoff_df: pd.DataFrame,
    biggest_leak: dict[str, Any],
    report_path: str | Path = Path("../output/funnel_report.json"),
    revenue_per_user: float | None = None,
) -> dict[str, Any]:
    """Write a structured JSON funnel report and print a console summary.

    Args:
        stages:           Stage label → count dict.
        dropoff_df:       Output of compute_dropoff (optionally with revenue columns).
        biggest_leak:     Output of identify_biggest_leak.
        report_path:      Where to save the JSON report.
        revenue_per_user: Optional; included in the report header for context.

    Returns:
        The report dict.
    """
    top_count = list(stages.values())[0] if stages else 1
    bottom_count = list(stages.values())[-1] if stages else 0
    overall_conversion = round(bottom_count / top_count * 100, 1) if top_count > 0 else 0.0

    report: dict[str, Any] = {
        "timestamp":           datetime.now().isoformat(),
        "overall_conversion_%": overall_conversion,
        "revenue_per_user":    revenue_per_user,
        "stages":              stages,
        "dropoff_transitions": dropoff_df.to_dict(orient="records"),
        "biggest_leak":        biggest_leak,
    }

    out_path = report_path if isinstance(report_path, Path) else Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print_funnel_summary(report)
    print(f"\n✓ Funnel report saved to {out_path}")

    return report


def _print_funnel_summary(report: dict[str, Any]) -> None:
    """Print a human-readable console summary of the funnel analysis."""
    print("\nFUNNEL ANALYSIS SUMMARY")
    print(f"Overall conversion: {report['overall_conversion_%']}%")

    print("\n── Stage counts ──")
    for stage, count in report["stages"].items():
        print(f"  {stage}: {count:,}")

    print("\n── Drop-off by transition (worst first) ──")
    for row in report["dropoff_transitions"]:
        flag = "  ⚠ BIGGEST LEAK" if (
            row["from_stage"] == report["biggest_leak"]["from_stage"]
            and row["to_stage"] == report["biggest_leak"]["to_stage"]
        ) else ""
        print(
            f"  {row['from_stage']} → {row['to_stage']}: "
            f"{row['drop_rate_%']}% drop ({row['users_lost']:,} lost){flag}"
        )

    leak = report["biggest_leak"]
    print(f"\n── Priority action ──")
    print(f"  {leak['recommendation']}")


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def run_funnel_analysis(
    df: pd.DataFrame,
    stage_definitions: dict[str, dict[str, Any]],
    revenue_per_user: float | None = None,
    output_dir: str | Path = Path("../output"),
    report_path: str | Path = Path("../output/funnel_report.json"),
    save_plots: bool = True,
) -> dict[str, Any]:
    """Run the complete funnel analysis pipeline in one call.

    Steps:
    1. Build stage counts from the DataFrame.
    2. Compute step-by-step drop-off metrics.
    3. Identify the biggest leak.
    4. Optionally calculate revenue impact.
    5. Generate and save charts.
    6. Write and return the JSON report.

    Args:
        df:                DataFrame with one row per user.
        stage_definitions: Ordered stage spec (see build_funnel_stages).
        revenue_per_user:  Average revenue per converted user (optional).
        output_dir:        Directory for chart PNGs.
        report_path:       Path for the JSON report.
        save_plots:        If True, saves charts to output_dir; else displays them.

    Returns:
        The complete funnel report dict.
    """
    out_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    # 1. Stages
    stages = build_funnel_stages(df, stage_definitions)

    # 2. Drop-off
    dropoff_df = compute_dropoff(stages)

    # 3. Biggest leak
    biggest_leak = identify_biggest_leak(dropoff_df)

    # 4. Revenue impact (optional)
    if revenue_per_user is not None:
        dropoff_df = calculate_business_impact(dropoff_df, revenue_per_user)

    # 5. Visualisations
    funnel_chart_path = (out_dir / "funnel_bar_chart.png") if save_plots else None
    dropoff_chart_path = (out_dir / "funnel_dropoff_rates.png") if save_plots else None
    plot_funnel(stages, output_path=funnel_chart_path)
    plot_dropoff_rates(dropoff_df, output_path=dropoff_chart_path)

    # 6. Report
    return generate_funnel_report(
        stages=stages,
        dropoff_df=dropoff_df,
        biggest_leak=biggest_leak,
        report_path=report_path,
        revenue_per_user=revenue_per_user,
    )
