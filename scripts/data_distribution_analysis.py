from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Non-interactive backend — safe for scripts that run without a display
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Step 1: Statistical summary with business interpretation
# ---------------------------------------------------------------------------

def compute_distribution_stats(
    series: pd.Series,
    column: str,
) -> dict:
    """Compute skewness, kurtosis, and descriptive stats with business interpretation.

    Skewness — is the distribution symmetric or pulled to one side?
      = 0       : symmetric (mean ≈ median, mean is reliable)
      > 1       : positive skew — long right tail — few very large values
                  (most customers are small; a few are huge → use median)
      < -1      : negative skew — long left tail — few very small values

    Kurtosis (excess) — how heavy are the tails vs a normal distribution?
      ≈ 0       : normal tails
      > 3       : fat tails — extreme outliers are likely
      < 0       : thin tails — values concentrated near the mean

    Args:
        series: Numeric pandas Series (NaNs are dropped before calculation).
        column: Column name used in the report and printout.

    Returns:
        Dict with all statistics and a plain-English business_interpretation.
    """
    clean = series.dropna()

    skewness = float(stats.skew(clean))
    kurtosis = float(stats.kurtosis(clean))   # excess kurtosis (normal = 0)
    mean = float(clean.mean())
    median = float(clean.median())
    std = float(clean.std())
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))

    # Plain-English interpretation connecting statistics to business meaning
    interpretation_parts: list[str] = []

    if abs(skewness) <= 0.5:
        interpretation_parts.append(
            f"Distribution is approximately symmetric (skew={skewness:.2f}). "
            f"Mean ({mean:.2f}) and median ({median:.2f}) are close — "
            f"mean is a reliable central measure."
        )
    elif skewness > 1:
        interpretation_parts.append(
            f"Highly right-skewed (skew={skewness:.2f}): most values are small "
            f"but a few large values pull the mean ({mean:.2f}) well above the "
            f"median ({median:.2f}). Use median for representative reporting."
        )
    elif skewness > 0.5:
        interpretation_parts.append(
            f"Moderately right-skewed (skew={skewness:.2f}): mean ({mean:.2f}) "
            f"is somewhat inflated by higher values. Consider median ({median:.2f})."
        )
    elif skewness < -1:
        interpretation_parts.append(
            f"Highly left-skewed (skew={skewness:.2f}): a few very small values "
            f"pull the mean ({mean:.2f}) below the median ({median:.2f}). "
            f"Use median for representative reporting."
        )
    else:
        interpretation_parts.append(
            f"Moderately left-skewed (skew={skewness:.2f}): "
            f"mean={mean:.2f}, median={median:.2f}."
        )

    if kurtosis > 3:
        interpretation_parts.append(
            f"Heavy tails (excess kurtosis={kurtosis:.2f}): extreme outliers "
            f"are likely — inspect and cap before modelling."
        )
    elif kurtosis < 0:
        interpretation_parts.append(
            f"Thin tails (excess kurtosis={kurtosis:.2f}): values are concentrated "
            f"near the mean — outlier risk is low."
        )

    interpretation = " ".join(interpretation_parts)
    print(f"  [{column}]  skew={skewness:.2f}  kurt={kurtosis:.2f}  "
          f"mean={mean:.2f}  median={median:.2f}")
    print(f"    → {interpretation}")

    return {
        "column": column,
        "n": int(len(clean)),
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "skewness": round(skewness, 4),
        "kurtosis_excess": round(kurtosis, 4),
        "business_interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Step 2: Histogram + KDE plot
# ---------------------------------------------------------------------------

def plot_distribution(
    series: pd.Series,
    column: str,
    output_dir: Path,
    bins: int = 30,
) -> str:
    """Plot a histogram with overlaid KDE curve and save to output/.

    Histogram shows the bucketed shape — clusters and gaps are visible.
    KDE (kernel density estimate) smooths the histogram to reveal the
    true continuous shape without bin-width artefacts.

    Side by side they answer: "What does the shape of this data look like?"

    Args:
        series: Numeric Series to plot.
        column: Column name — used in title and filename.
        output_dir: Directory to save the PNG.
        bins: Number of histogram bins (default 30).

    Returns:
        Path string of the saved PNG file.
    """
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Histogram — shows bucketed shape
    axes[0].hist(clean, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{column} — Histogram")
    axes[0].axvline(clean.mean(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Mean {clean.mean():.1f}")
    axes[0].axvline(clean.median(), color="orange", linestyle="-.", linewidth=1.2,
                    label=f"Median {clean.median():.1f}")
    axes[0].legend(fontsize=8)

    # Right: KDE — smooth continuous shape, easier to see true distribution
    clean.plot(kind="density", ax=axes[1], color="steelblue", linewidth=2)
    axes[1].set_xlabel(column)
    axes[1].set_title(f"{column} — KDE (Smoothed)")
    axes[1].axvline(clean.mean(), color="red", linestyle="--", linewidth=1.2,
                    label=f"Mean {clean.mean():.1f}")
    axes[1].axvline(clean.median(), color="orange", linestyle="-.", linewidth=1.2,
                    label=f"Median {clean.median():.1f}")
    axes[1].legend(fontsize=8)

    plt.suptitle(f"Distribution Analysis: {column}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / f"distribution_{column}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Distribution plot saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Step 3: Segment comparison
# ---------------------------------------------------------------------------

def plot_segment_comparison(
    df: pd.DataFrame,
    column: str,
    segment_column: str,
    output_dir: Path,
    bins: int = 30,
) -> str:
    """Overlay histogram distributions across segments to reveal group differences.

    Segment comparison answers: "Do high-value customers behave differently
    from low-value ones?"  Overlapping distributions suggest no difference;
    separated distributions confirm distinct business segments that may
    warrant different strategies.

    Args:
        df: Input DataFrame.
        column: Numeric column to compare across segments.
        segment_column: Categorical column defining the segments.
        output_dir: Directory to save the PNG.
        bins: Number of histogram bins.

    Returns:
        Path string of the saved PNG file.
    """
    if column not in df.columns or segment_column not in df.columns:
        print(f"  ⚠  Segment comparison skipped — column missing")
        return ""

    segments = df[segment_column].dropna().unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    colours = plt.cm.tab10.colors
    for i, seg in enumerate(segments):
        subset = df.loc[df[segment_column] == seg, column].dropna()
        ax.hist(subset, bins=bins, alpha=0.55, label=str(seg),
                color=colours[i % len(colours)], edgecolor="white")

    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"{column} distribution by {segment_column}")
    ax.legend(title=segment_column, fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"segment_{column}_by_{segment_column}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Segment comparison plot saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Step 4: High-value vs low-value quartile comparison (lesson pattern)
# ---------------------------------------------------------------------------

def compare_high_low_segments(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    bins: int = 30,
) -> dict:
    """Split column into top and bottom quartile and compare distributions.

    This is the exact pattern from HRS2.28:
        high_value = df[df[col] > df[col].quantile(0.75)]
        low_value  = df[df[col] < df[col].quantile(0.25)]

    The comparison reveals whether extreme segments have fundamentally
    different shapes — if they do, they likely represent different business
    types requiring different treatment.

    Args:
        df: Input DataFrame.
        column: Numeric column to split.
        output_dir: Directory to save the PNG.
        bins: Number of histogram bins.

    Returns:
        Dict with q25, q75, and plot path.
    """
    if column not in df.columns:
        return {"status": "skipped", "reason": f"Column '{column}' not found."}

    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)

    high_value = df[df[column] > q75][column].dropna()
    low_value = df[df[column] < q25][column].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(high_value, alpha=0.6, label=f"High-Value (>{q75:.1f})",
            bins=bins, color="steelblue", edgecolor="white")
    ax.hist(low_value, alpha=0.6, label=f"Low-Value (<{q25:.1f})",
            bins=bins, color="coral", edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(f"{column} — High vs Low Quartile Comparison")
    ax.legend(fontsize=9)
    plt.tight_layout()

    output_path = output_dir / f"highlow_{column}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ High/low segment plot saved → {output_path}  "
          f"(high n={len(high_value)}, low n={len(low_value)})")

    return {
        "column": column,
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "high_value_n": int(len(high_value)),
        "low_value_n": int(len(low_value)),
        "plot": str(output_path),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_distribution_analysis(
    df: pd.DataFrame,
    columns: list[str],
    output_dir: str | Path,
    segment_comparisons: list[dict] | None = None,
) -> dict:
    """Run the full distribution analysis pipeline for multiple columns.

    For each column:
      1. Compute skewness, kurtosis, descriptive stats + business interpretation
      2. Plot histogram + KDE — shape visible
      3. Plot high vs low quartile comparison — segment differences visible

    Optionally, overlay histograms across a categorical segment column.

    Args:
        df: Input DataFrame (post processing).
        columns: Numeric columns to analyse.
        output_dir: Directory for saving PNG plots and JSON report.
        segment_comparisons: List of {column, segment_column} dicts for
                             categorical segment overlay plots.

    Returns:
        Full distribution analysis report dict.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    segment_comparisons = segment_comparisons or []
    column_reports: list[dict] = []
    highlow_reports: list[dict] = []
    plots: list[str] = []

    print("\n── Distribution Analysis for Business Trends ─────────────────")

    for col in columns:
        if col not in df.columns:
            print(f"  ⚠  [{col}] not found — skipped")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  ⚠  [{col}] not numeric — skipped")
            continue

        print(f"\n  Analysing: {col}")

        # Step 1: Statistics + interpretation
        stats_report = compute_distribution_stats(df[col], col)
        column_reports.append(stats_report)

        # Step 2: Histogram + KDE
        plot_path = plot_distribution(df[col], col, out)
        plots.append(plot_path)

        # Step 3: High vs low quartile comparison
        hl = compare_high_low_segments(df, col, out)
        highlow_reports.append(hl)
        if "plot" in hl:
            plots.append(hl["plot"])

    # Optional segment comparison plots
    for sc in segment_comparisons:
        col = sc.get("column", "")
        seg_col = sc.get("segment_column", "")
        if col and seg_col:
            print(f"\n  Segment comparison: {col} by {seg_col}")
            p = plot_segment_comparison(df, col, seg_col, out)
            if p:
                plots.append(p)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Columns analysed : {len(column_reports)}")
    print(f"  Plots saved      : {len(plots)}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "columns_analysed": len(column_reports),
        "plots_saved": plots,
        "statistics": column_reports,
        "high_low_comparisons": highlow_reports,
    }

    return report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_distribution_report(report: dict, output_path: str | Path) -> None:
    """Persist the distribution analysis report to JSON.

    Args:
        report: Report dict returned by run_distribution_analysis().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Distribution analysis report saved → {path}")
