from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Narrow time: isolate when the anomaly occurred
# ---------------------------------------------------------------------------

def isolate_anomaly_window(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str = "date",
    freq: str = "D",
    anomaly_threshold: float | None = None,
    std_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Resample a metric by time frequency and detect anomalous periods.

    An anomaly period is any bucket where the metric falls below
    (mean - std_multiplier * std).  If anomaly_threshold is provided
    explicitly it is used instead of the statistical threshold.

    Args:
        df:                DataFrame with a datetime date_col.
        metric_col:        Numeric column to analyse (e.g. 'amount', 'success_rate').
        date_col:          Name of the datetime column.
        freq:              Resample frequency: 'D' daily, 'h' hourly, 'W' weekly.
        anomaly_threshold: Explicit threshold below which a bucket is anomalous.
                           If None, uses mean - std_multiplier * std.
        std_multiplier:    How many standard deviations below mean counts as anomaly.

    Returns:
        Dict with keys:
        - resampled  : Series of metric values per bucket
        - threshold  : The computed or provided threshold value
        - anomaly_periods: list of period labels where metric < threshold
        - worst_period : the single lowest-metric period
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")
    if metric_col not in df.columns:
        raise KeyError(f"Metric column '{metric_col}' not found in DataFrame.")

    resampled = df.set_index(date_col)[metric_col].resample(freq).mean()
    resampled = resampled.dropna()

    if anomaly_threshold is None:
        threshold = float(resampled.mean() - std_multiplier * resampled.std())
    else:
        threshold = float(anomaly_threshold)

    anomaly_mask    = resampled < threshold
    anomaly_periods = [str(idx) for idx in resampled[anomaly_mask].index.tolist()]
    worst_period    = str(resampled.idxmin()) if not resampled.empty else None

    return {
        "resampled":       resampled,
        "threshold":       round(threshold, 4),
        "anomaly_periods": anomaly_periods,
        "worst_period":    worst_period,
        "freq":            freq,
        "metric_col":      metric_col,
    }


def drill_into_period(
    df: pd.DataFrame,
    period: str,
    date_col: str = "date",
    drill_freq: str = "h",
) -> pd.Series:
    """Zoom into a single day or period at a finer time granularity.

    Useful for identifying the exact hour-range of an incident once the
    anomalous day is known.

    Args:
        df:          DataFrame with datetime date_col.
        period:      Period string to filter on (e.g. '2024-01-15').
                     Matches any row where date_col starts with this string.
        date_col:    Name of the datetime column.
        drill_freq:  Finer resample frequency for the zoomed view.

    Returns:
        Series of mean metric values at drill_freq granularity for that period.
    """
    mask   = df[date_col].dt.strftime("%Y-%m-%d") == period[:10]
    subset = df[mask]
    if subset.empty:
        return pd.Series(dtype=float)

    numeric_cols = subset.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.Series(dtype=float)

    return subset.set_index(date_col)[numeric_cols[0]].resample(drill_freq).mean()


# ---------------------------------------------------------------------------
# Step 2 — Narrow segment: find which slice is responsible
# ---------------------------------------------------------------------------

def segment_metric_during_anomaly(
    df: pd.DataFrame,
    metric_col: str,
    segment_col: str,
    anomaly_periods: list[str],
    date_col: str = "date",
    freq: str = "D",
    agg: str = "mean",
) -> pd.DataFrame:
    """Compare a metric across segment values during vs. outside anomaly periods.

    This surfaces which segment (payment method, region, product, etc.)
    drives the anomalous behaviour.

    Args:
        df:               DataFrame with datetime date_col.
        metric_col:       Numeric column to aggregate.
        segment_col:      Categorical column to split by (e.g. 'region').
        anomaly_periods:  Output of isolate_anomaly_window['anomaly_periods'].
        date_col:         Name of the datetime column.
        freq:             Time bucket used when matching anomaly_periods.
        agg:              Aggregation: 'mean', 'sum', or 'count'.

    Returns:
        DataFrame with columns: segment_value, during_anomaly, outside_anomaly,
        absolute_diff, relative_diff_% — sorted by absolute_diff ascending
        (most affected segment first).
    """
    if segment_col not in df.columns:
        raise KeyError(f"Segment column '{segment_col}' not found in DataFrame.")

    # Tag each row as inside or outside the anomaly window
    period_labels = df[date_col].dt.to_period(freq).astype(str)
    in_anomaly    = period_labels.isin(anomaly_periods)

    AGG_FN = {"mean": "mean", "sum": "sum", "count": "count"}
    fn = AGG_FN.get(agg, "mean")

    during  = df[in_anomaly].groupby(segment_col)[metric_col].agg(fn)
    outside = df[~in_anomaly].groupby(segment_col)[metric_col].agg(fn)

    comparison = pd.DataFrame({
        "during_anomaly":  during,
        "outside_anomaly": outside,
    }).fillna(0)

    comparison["absolute_diff"]    = comparison["during_anomaly"] - comparison["outside_anomaly"]
    comparison["relative_diff_%"]  = (
        comparison["absolute_diff"] / comparison["outside_anomaly"].replace(0, float("nan")) * 100
    ).round(1)

    return (
        comparison
        .reset_index()
        .rename(columns={segment_col: "segment_value"})
        .sort_values("absolute_diff")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Step 3 — Find pattern: correlate the anomaly across multiple dimensions
# ---------------------------------------------------------------------------

def correlate_with_segments(
    df: pd.DataFrame,
    metric_col: str,
    segment_cols: list[str],
    anomaly_periods: list[str],
    date_col: str = "date",
    freq: str = "D",
) -> dict[str, pd.DataFrame]:
    """Run segment_metric_during_anomaly across multiple segment columns.

    Produces one comparison table per segment, giving a multi-dimensional
    view of which variables correlate with the anomaly.

    Args:
        df:              DataFrame with datetime date_col.
        metric_col:      Metric to compare.
        segment_cols:    List of columns to segment by.
        anomaly_periods: Output of isolate_anomaly_window['anomaly_periods'].
        date_col:        Name of the datetime column.
        freq:            Time bucket frequency.

    Returns:
        Dict mapping segment_col → comparison DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    for col in segment_cols:
        if col not in df.columns:
            print(f"  ⚠ Skipping segment '{col}' — column not found.")
            continue
        results[col] = segment_metric_during_anomaly(
            df, metric_col, col, anomaly_periods, date_col, freq
        )
    return results


# ---------------------------------------------------------------------------
# Step 4 — Hypothesis builder: structure findings into a formal report
# ---------------------------------------------------------------------------

def build_hypothesis(
    observation: str,
    narrowed_time: str,
    narrowed_segment: str,
    correlated_patterns: list[str],
    root_cause: str,
    supporting_evidence: list[str],
    recommended_action: str,
) -> dict[str, Any]:
    """Construct a structured root cause hypothesis from investigation findings.

    All fields are plain-English strings written by the analyst based on
    the evidence surfaced by the previous three steps.

    Args:
        observation:         The original anomaly that triggered investigation.
        narrowed_time:       When the anomaly occurred (from isolate_anomaly_window).
        narrowed_segment:    Which segment was most affected (from segment analysis).
        correlated_patterns: List of patterns that correlate with the anomaly.
        root_cause:          Single-sentence root cause hypothesis.
        supporting_evidence: List of evidence items supporting the hypothesis.
        recommended_action:  Concrete next action to resolve the root cause.

    Returns:
        Structured hypothesis dict ready for inclusion in the RCA report.
    """
    return {
        "observation":         observation,
        "narrowed_time":       narrowed_time,
        "narrowed_segment":    narrowed_segment,
        "correlated_patterns": correlated_patterns,
        "root_cause":          root_cause,
        "supporting_evidence": supporting_evidence,
        "recommended_action":  recommended_action,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_metric_timeline(
    resampled: pd.Series,
    threshold: float,
    anomaly_periods: list[str],
    title: str = "Metric Timeline with Anomaly Detection",
    output_path: str | Path | None = None,
) -> None:
    """Plot metric over time with anomaly threshold and highlighted anomaly periods.

    Args:
        resampled:       Time-indexed Series (output of isolate_anomaly_window).
        threshold:       Threshold line value.
        anomaly_periods: Periods to shade red.
        title:           Chart title.
        output_path:     If provided, saves as PNG; otherwise displays.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(resampled.index, resampled.values, color="#3b82f6", linewidth=2, label="Metric")
    ax.axhline(threshold, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Threshold ({threshold:.2f})")

    # Shade anomaly buckets
    for period_str in anomaly_periods:
        try:
            period_ts = pd.Timestamp(period_str[:10])
            ax.axvspan(
                period_ts, period_ts + pd.Timedelta(days=1),
                color="#fca5a5", alpha=0.4, label="_nolegend_",
            )
        except Exception:
            pass

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(resampled.name or "Value")
    ax.set_xlabel("Date")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    except Exception:
        pass

    plt.tight_layout()

    if output_path:
        out = output_path if isinstance(output_path, Path) else Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ Timeline chart saved to {out}")
        plt.close(fig)
    else:
        plt.show()


def plot_segment_comparison(
    comparison_df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    output_path: str | Path | None = None,
) -> None:
    """Side-by-side bar chart of metric during vs. outside the anomaly per segment.

    The most-affected segment (largest absolute drop) is always at the top
    so the root cause candidate is immediately visible.

    Args:
        comparison_df: Output of segment_metric_during_anomaly.
        segment_col:   Column name used as the segment dimension (for labels).
        metric_col:    Metric name (for axis label).
        output_path:   If provided, saves as PNG; otherwise displays.
    """
    if comparison_df.empty:
        print(f"  ⚠ No data to plot for segment '{segment_col}'.")
        return

    labels  = comparison_df["segment_value"].astype(str).tolist()
    during  = comparison_df["during_anomaly"].tolist()
    outside = comparison_df["outside_anomaly"].tolist()
    n       = len(labels)

    x     = range(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    ax.bar([i - width / 2 for i in x], outside, width, label="Outside anomaly", color="#3b82f6", alpha=0.85)
    ax.bar([i + width / 2 for i in x], during,  width, label="During anomaly",  color="#ef4444", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(f"'{metric_col}' by '{segment_col}': During vs. Outside Anomaly",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_path:
        out = output_path if isinstance(output_path, Path) else Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ Segment comparison chart saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_rca_report(
    anomaly_result: dict[str, Any],
    segment_correlations: dict[str, pd.DataFrame],
    hypothesis: dict[str, Any],
    report_path: str | Path = Path("../output/rca_report.json"),
) -> dict[str, Any]:
    """Write a structured JSON Root Cause Analysis report and print a summary.

    The report captures the full investigation trail:
    - Anomaly time window and worst period
    - Segment comparison tables for each dimension analysed
    - Structured hypothesis with evidence and recommended action

    Args:
        anomaly_result:       Output of isolate_anomaly_window.
        segment_correlations: Output of correlate_with_segments.
        hypothesis:           Output of build_hypothesis.
        report_path:          Where to save the JSON report.

    Returns:
        The report dict.
    """
    # Serialise segment DataFrames to records
    serialised_segments: dict[str, list[dict]] = {
        col: df.to_dict(orient="records")
        for col, df in segment_correlations.items()
    }

    report: dict[str, Any] = {
        "timestamp":      datetime.now().isoformat(),
        "metric_analysed": anomaly_result.get("metric_col"),
        "anomaly_window": {
            "freq":            anomaly_result.get("freq"),
            "threshold":       anomaly_result.get("threshold"),
            "anomaly_periods": anomaly_result.get("anomaly_periods", []),
            "worst_period":    anomaly_result.get("worst_period"),
        },
        "segment_correlations": serialised_segments,
        "hypothesis": hypothesis,
    }

    out_path = report_path if isinstance(report_path, Path) else Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print_rca_summary(report)
    print(f"\n✓ RCA report saved to {out_path}")

    return report


def _print_rca_summary(report: dict[str, Any]) -> None:
    """Print a structured investigation summary to the console."""
    h = report["hypothesis"]
    w = report["anomaly_window"]

    print("\nROOT CAUSE ANALYSIS SUMMARY")
    print(f"Metric:         {report['metric_analysed']}")
    print(f"Worst period:   {w['worst_period']}")
    print(f"Anomaly periods ({len(w['anomaly_periods'])}): {', '.join(w['anomaly_periods'][:5])}"
          + (" ..." if len(w["anomaly_periods"]) > 5 else ""))

    print(f"\nObservation:      {h['observation']}")
    print(f"Narrowed time:    {h['narrowed_time']}")
    print(f"Narrowed segment: {h['narrowed_segment']}")

    print("\nCorrelated patterns:")
    for pattern in h["correlated_patterns"]:
        print(f"  - {pattern}")

    print(f"\nRoot cause:  {h['root_cause']}")

    print("\nSupporting evidence:")
    for item in h["supporting_evidence"]:
        print(f"  • {item}")

    print(f"\nRecommended action: {h['recommended_action']}")

    if report["segment_correlations"]:
        print("\n── Segment impact summary ──")
        for seg_col, rows in report["segment_correlations"].items():
            if rows:
                worst = rows[0]  # sorted ascending by absolute_diff
                print(
                    f"  {seg_col}: most affected = '{worst['segment_value']}' "
                    f"(during: {worst['during_anomaly']:.2f} vs "
                    f"outside: {worst['outside_anomaly']:.2f}, "
                    f"diff: {worst['relative_diff_%']}%)"
                )


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def run_rca(
    df: pd.DataFrame,
    metric_col: str,
    segment_cols: list[str],
    hypothesis: dict[str, Any],
    date_col: str = "date",
    freq: str = "D",
    anomaly_threshold: float | None = None,
    std_multiplier: float = 2.0,
    output_dir: str | Path = Path("../output"),
    report_path: str | Path = Path("../output/rca_report.json"),
    save_plots: bool = True,
) -> dict[str, Any]:
    """Run the complete root cause investigation pipeline in one call.

    Steps:
    1. Isolate the anomaly time window.
    2. Correlate the metric across all provided segment columns.
    3. Save the timeline and per-segment comparison charts.
    4. Write the JSON report and return it.

    Args:
        df:                DataFrame with datetime date_col.
        metric_col:        Metric column to investigate.
        segment_cols:      Categorical columns to segment by.
        hypothesis:        Pre-built hypothesis dict (from build_hypothesis).
                           The analyst populates this after reviewing the charts.
        date_col:          Name of the datetime column.
        freq:              Time bucket frequency for anomaly detection.
        anomaly_threshold: Explicit threshold; if None uses statistical rule.
        std_multiplier:    Std deviations below mean = anomaly (when threshold=None).
        output_dir:        Directory for chart PNGs.
        report_path:       Path for JSON report.
        save_plots:        Save charts if True; display otherwise.

    Returns:
        The complete RCA report dict.
    """
    out_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)

    # Step 1: Time isolation
    anomaly_result = isolate_anomaly_window(
        df, metric_col, date_col, freq, anomaly_threshold, std_multiplier
    )

    # Step 2: Segment correlation
    segment_correlations = correlate_with_segments(
        df, metric_col, segment_cols, anomaly_result["anomaly_periods"], date_col, freq
    )

    # Step 3: Charts
    if save_plots:
        plot_metric_timeline(
            anomaly_result["resampled"],
            anomaly_result["threshold"],
            anomaly_result["anomaly_periods"],
            title=f"'{metric_col}' Timeline — Anomaly Detection",
            output_path=out_dir / "rca_timeline.png",
        )
        for seg_col, comp_df in segment_correlations.items():
            safe_name = seg_col.replace(" ", "_")
            plot_segment_comparison(
                comp_df, seg_col, metric_col,
                output_path=out_dir / f"rca_segment_{safe_name}.png",
            )
    else:
        plot_metric_timeline(
            anomaly_result["resampled"],
            anomaly_result["threshold"],
            anomaly_result["anomaly_periods"],
        )
        for seg_col, comp_df in segment_correlations.items():
            plot_segment_comparison(comp_df, seg_col, metric_col)

    # Step 4: Report
    return generate_rca_report(
        anomaly_result=anomaly_result,
        segment_correlations=segment_correlations,
        hypothesis=hypothesis,
        report_path=report_path,
    )
