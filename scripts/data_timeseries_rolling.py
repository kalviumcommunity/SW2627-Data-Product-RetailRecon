from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Non-interactive backend — safe for scripts running without a display
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Step 1: Rolling averages — smooth noise to reveal true trend
# ---------------------------------------------------------------------------

def compute_rolling_metrics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling window averages that smooth daily noise to reveal trend.

    Raw daily values are maximum detail but maximum noise.
    A 7-day rolling average smooths the spiky daily signal and reveals
    the sustainable trend underneath — whether business is accelerating,
    stable, or declining.

    Rolling windows work on row order — the DataFrame must be sorted by
    date before calling this function.

    Windows:
      7  — 1-week smoothing: removes daily day-of-week pattern
      30 — 1-month smoothing: removes weekly volatility, shows monthly trend

    min_periods=1 means results are available from the first row (no NaN
    at the start of the series for partial windows).

    Args:
        df: DataFrame sorted by date_col.
        date_col: Datetime column to sort and index on.
        value_col: Numeric column to compute rolling metrics on.
        windows: List of window sizes in rows (default [7, 30]).

    Returns:
        DataFrame with new {value_col}_ma{window} columns appended.
    """
    windows = windows or [7, 30]
    working_df = df.copy().sort_values(date_col).reset_index(drop=True)

    for w in windows:
        col_name = f"{value_col}_ma{w}"
        # rolling(window).mean() — sliding window average, min_periods=1
        # avoids leading NaN when fewer than window rows are available
        working_df[col_name] = (
            working_df[value_col]
            .rolling(window=w, min_periods=1)
            .mean()
        )
        print(f"  ✓ [{col_name}] rolling mean, window={w}")

    return working_df


# ---------------------------------------------------------------------------
# Step 2: Resample by time period
# ---------------------------------------------------------------------------

def resample_by_period(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "W",
    agg_funcs: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a column by calendar period using resample.

    Resampling aggregates all rows within each period to a single value —
    one row per week, one per month, one per quarter.  This removes all
    intra-period detail but makes the big-picture direction clear.

    Common frequency aliases:
      'D'  = calendar day     'W'  = week end
      'ME' = month end        'QE' = quarter end
      'YE' = year end         'h'  = hour

    Args:
        df: DataFrame with a parsed datetime column.
        date_col: Datetime column to use as the time index.
        value_col: Numeric column to aggregate.
        freq: Resample frequency alias (default 'W' = weekly).
        agg_funcs: Aggregation functions to compute (default ['sum', 'mean', 'count']).

    Returns:
        Resampled DataFrame with datetime index reset to column.
    """
    agg_funcs = agg_funcs or ["sum", "mean", "count"]

    if date_col not in df.columns or value_col not in df.columns:
        print(f"  ⚠  Resample skipped — '{date_col}' or '{value_col}' not found")
        return pd.DataFrame()

    # Set datetime as index — required for .resample() to work
    df_ts = df.set_index(date_col)[[value_col]].copy()

    agg_dict = {func: (value_col, func) for func in agg_funcs}
    resampled = df_ts[value_col].resample(freq).agg(agg_funcs)
    resampled.columns = [f"{value_col}_{func}" for func in agg_funcs]
    resampled = resampled.reset_index()

    print(f"  ✓ Resampled '{value_col}' by '{freq}' "
          f"({agg_funcs}) → {len(resampled)} period(s)")

    return resampled


# ---------------------------------------------------------------------------
# Step 3: Period-over-period change
# ---------------------------------------------------------------------------

def compute_period_changes(
    resampled_df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute period-over-period percentage change using pct_change().

    pct_change() measures growth rate: how much did the metric change
    relative to the previous period?
      +10% = growing at 10% per period (momentum)
      -5%  = declining at 5% per period (warning signal)
      0%   = flat — no growth, no decline

    This is the "business momentum" view — trend direction in one number.

    Args:
        resampled_df: Output of resample_by_period().
        date_col: Datetime column in the resampled DataFrame.
        value_col: The resampled metric column (e.g. 'amount_sum').

    Returns:
        DataFrame with added {value_col}_pct_change column.
    """
    if value_col not in resampled_df.columns:
        print(f"  ⚠  pct_change skipped — '{value_col}' not found")
        return resampled_df

    working_df = resampled_df.copy()
    # pct_change() * 100 → percentage, first row is NaN (no prior period)
    working_df[f"{value_col}_pct_change"] = (
        working_df[value_col].pct_change() * 100
    )

    latest_change = working_df[f"{value_col}_pct_change"].iloc[-1]
    if not pd.isna(latest_change):
        direction = "▲" if latest_change > 0 else ("▼" if latest_change < 0 else "→")
        print(f"  ✓ [{value_col}_pct_change]  "
              f"latest period-over-period: {direction} {latest_change:+.1f}%")

    return working_df


# ---------------------------------------------------------------------------
# Step 4: Cumulative sum
# ---------------------------------------------------------------------------

def compute_cumulative_sum(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Compute running cumulative total — how much has accumulated to each point.

    cumsum() answers: "What is our total revenue from the beginning of the
    period to this date?"  It turns a daily flow into a growing stock and
    reveals the overall trajectory of a business metric.

    Args:
        df: DataFrame sorted by date_col.
        date_col: Datetime column (used for sorting only).
        value_col: Numeric column to accumulate.

    Returns:
        DataFrame with {value_col}_cumulative column appended.
    """
    working_df = df.copy().sort_values(date_col).reset_index(drop=True)
    out_col = f"{value_col}_cumulative"
    working_df[out_col] = working_df[value_col].cumsum()

    total = working_df[out_col].iloc[-1]
    print(f"  ✓ [{out_col}]  running total at last record: {total:.2f}")

    return working_df


# ---------------------------------------------------------------------------
# Step 5: Trend direction identification
# ---------------------------------------------------------------------------

def identify_trend(
    resampled_df: pd.DataFrame,
    value_col: str,
    lookback_periods: int = 3,
) -> dict:
    """Identify trend direction by comparing latest period to N periods ago.

    Trend logic from the lesson:
      latest > latest - N periods → Uptrend (Accelerating)
      latest < latest - N periods → Downtrend (Declining)
      latest ≈ latest - N periods → Flat (Stable)

    Args:
        resampled_df: Resampled DataFrame with the metric column.
        value_col: Column to measure trend on.
        lookback_periods: How many periods back to compare (default 3).

    Returns:
        Dict with trend direction, delta, and business implication string.
    """
    if value_col not in resampled_df.columns or len(resampled_df) < 2:
        return {"trend": "insufficient_data", "delta": None, "implication": ""}

    latest = resampled_df[value_col].iloc[-1]
    lookback_idx = max(0, len(resampled_df) - 1 - lookback_periods)
    comparison = resampled_df[value_col].iloc[lookback_idx]

    delta = latest - comparison
    pct = ((delta / comparison) * 100) if comparison != 0 else 0.0

    if delta > 0:
        trend = "uptrend"
        implication = (
            f"Business is ACCELERATING: {value_col} grew by "
            f"{delta:.2f} ({pct:+.1f}%) over the last {lookback_periods} period(s). "
            f"Monitor to confirm sustained momentum."
        )
    elif delta < 0:
        trend = "downtrend"
        implication = (
            f"Business is DECLINING: {value_col} dropped by "
            f"{abs(delta):.2f} ({pct:+.1f}%) over the last {lookback_periods} period(s). "
            f"Investigate root cause before taking corrective action."
        )
    else:
        trend = "flat"
        implication = (
            f"Business is FLAT: {value_col} unchanged over "
            f"the last {lookback_periods} period(s). "
            f"Monitor for emerging direction."
        )

    print(f"  ✓ Trend: {trend.upper()}  ({pct:+.1f}% over {lookback_periods} period(s))")
    print(f"    → {implication}")

    return {
        "trend": trend,
        "latest_value": round(float(latest), 4),
        "comparison_value": round(float(comparison), 4),
        "delta": round(float(delta), 4),
        "pct_change": round(float(pct), 2),
        "lookback_periods": lookback_periods,
        "implication": implication,
    }


# ---------------------------------------------------------------------------
# Step 6: Visualisation — raw vs rolling + cumulative
# ---------------------------------------------------------------------------

def plot_rolling_trend(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    rolling_cols: list[str],
    output_dir: Path,
) -> str:
    """Plot raw values alongside rolling averages to show signal vs noise.

    Raw data (alpha=0.3) shows every spike and dip.
    Rolling average lines (solid) reveal the true underlying trend.
    The contrast between them is the visual lesson: rolling average
    filters noise so the trend becomes readable.

    Args:
        df: DataFrame with date, raw value, and rolling columns.
        date_col: Datetime column for x-axis.
        value_col: Raw metric column.
        rolling_cols: List of rolling average column names to overlay.
        output_dir: Directory to save the PNG.

    Returns:
        Path string of the saved PNG.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw data — high alpha for noise context
    ax.plot(df[date_col], df[value_col],
            alpha=0.3, color="steelblue", linewidth=1, label="Raw")

    colours = ["red", "orange", "green", "purple"]
    for i, col in enumerate(rolling_cols):
        if col in df.columns:
            ax.plot(df[date_col], df[col],
                    color=colours[i % len(colours)],
                    linewidth=2, label=col)

    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.set_title(
        f"{value_col} — Raw vs Rolling Average\n"
        f"(raw=noise, rolling=signal)"
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / f"rolling_trend_{value_col}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Rolling trend plot saved → {output_path}")
    return str(output_path)


def plot_period_over_period(
    resampled_df: pd.DataFrame,
    date_col: str,
    value_col: str,
    pct_change_col: str,
    output_dir: Path,
) -> str:
    """Two-panel plot: resampled totals (bar) + period-over-period % change (line).

    Top panel: aggregated value per period — absolute scale
    Bottom panel: pct_change — relative growth rate
    Together they show both magnitude and momentum.

    Args:
        resampled_df: Output of compute_period_changes().
        date_col: Datetime column for x-axis.
        value_col: Resampled metric column.
        pct_change_col: Percentage change column.
        output_dir: Directory to save the PNG.

    Returns:
        Path string of the saved PNG.
    """
    if value_col not in resampled_df.columns:
        return ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: bar chart of resampled totals
    ax1.bar(resampled_df[date_col].astype(str),
            resampled_df[value_col],
            color="steelblue", alpha=0.8, edgecolor="white")
    ax1.set_ylabel(value_col)
    ax1.set_title(f"{value_col} — Period Aggregation & Momentum")

    # Bottom: period-over-period % change
    if pct_change_col in resampled_df.columns:
        changes = resampled_df[pct_change_col].fillna(0)
        colours = ["green" if v >= 0 else "red" for v in changes]
        ax2.bar(resampled_df[date_col].astype(str),
                changes, color=colours, alpha=0.8, edgecolor="white")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("% Change vs Prior Period")
        ax2.set_xlabel("Period")

    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"period_change_{value_col}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Period-over-period plot saved → {output_path}")
    return str(output_path)


def plot_cumulative(
    df: pd.DataFrame,
    date_col: str,
    cumulative_col: str,
    output_dir: Path,
) -> str:
    """Plot cumulative sum over time — the growing total trajectory.

    Args:
        df: DataFrame with date and cumulative column.
        date_col: Datetime column for x-axis.
        cumulative_col: Cumulative sum column to plot.
        output_dir: Directory to save the PNG.

    Returns:
        Path string of the saved PNG.
    """
    if cumulative_col not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df[date_col], df[cumulative_col],
                    alpha=0.25, color="steelblue")
    ax.plot(df[date_col], df[cumulative_col],
            color="steelblue", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel(cumulative_col)
    ax.set_title(f"Cumulative Total: {cumulative_col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / f"cumulative_{cumulative_col}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Cumulative plot saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_timeseries_analysis(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    output_dir: str | Path,
    rolling_windows: list[int] | None = None,
    resample_freq: str = "W",
    trend_lookback: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Run the full time-series trend and rolling metrics pipeline.

    Steps:
      1. Rolling averages — smooth noise to reveal trend (7-day, 30-day)
      2. Cumulative sum   — running total trajectory
      3. Resample by period — one value per week/month
      4. Period-over-period change — growth rate / momentum
      5. Trend identification — uptrend / downtrend / flat + implication
      6. Plots — raw vs rolling, period change, cumulative

    Args:
        df: Processed DataFrame with a parsed datetime column.
        date_col: Datetime column name.
        value_col: Numeric column to analyse (e.g. 'amount').
        output_dir: Directory for saving plots and the JSON report.
        rolling_windows: Window sizes for rolling averages (default [7, 30]).
        resample_freq: Resample frequency alias (default 'W' for weekly).
        trend_lookback: Number of periods to look back for trend direction.

    Returns:
        (enriched DataFrame with rolling + cumulative columns, report dict)
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolling_windows = rolling_windows or [7, 30]
    plots: list[str] = []

    print("\n── Time-Series Trend & Rolling Metrics ────────────────────────")

    # Validate inputs
    if date_col not in df.columns:
        print(f"  ⚠  '{date_col}' not found — pipeline skipped")
        return df, {"status": "skipped", "reason": f"'{date_col}' not found"}
    if value_col not in df.columns:
        print(f"  ⚠  '{value_col}' not found — pipeline skipped")
        return df, {"status": "skipped", "reason": f"'{value_col}' not found"}

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # ── Step 1: Rolling averages ────────────────────────────────────────
    print(f"\n[1/5] Rolling averages  windows={rolling_windows}")
    df = compute_rolling_metrics(df, date_col, value_col, windows=rolling_windows)
    rolling_cols = [f"{value_col}_ma{w}" for w in rolling_windows]

    # ── Step 2: Cumulative sum ──────────────────────────────────────────
    print(f"\n[2/5] Cumulative sum")
    df = compute_cumulative_sum(df, date_col, value_col)

    # ── Step 3: Resample by period ──────────────────────────────────────
    print(f"\n[3/5] Resample by period  freq='{resample_freq}'")
    resampled = resample_by_period(
        df, date_col, value_col, freq=resample_freq,
        agg_funcs=["sum", "mean", "count"],
    )
    resampled_sum_col = f"{value_col}_sum"

    # ── Step 4: Period-over-period change ───────────────────────────────
    print(f"\n[4/5] Period-over-period change")
    if not resampled.empty and resampled_sum_col in resampled.columns:
        resampled = compute_period_changes(resampled, date_col, resampled_sum_col)
        pct_col = f"{resampled_sum_col}_pct_change"
    else:
        pct_col = ""

    # ── Step 5: Trend identification ────────────────────────────────────
    print(f"\n[5/5] Trend identification  lookback={trend_lookback}")
    trend_result: dict = {}
    if not resampled.empty and resampled_sum_col in resampled.columns:
        trend_result = identify_trend(
            resampled, resampled_sum_col, lookback_periods=trend_lookback
        )

    # ── Plots ────────────────────────────────────────────────────────────
    print(f"\n  Generating plots →")

    p = plot_rolling_trend(df, date_col, value_col, rolling_cols, out)
    if p:
        plots.append(p)

    if not resampled.empty:
        p = plot_period_over_period(
            resampled, date_col, resampled_sum_col, pct_col, out
        )
        if p:
            plots.append(p)

    cumulative_col = f"{value_col}_cumulative"
    p = plot_cumulative(df, date_col, cumulative_col, out)
    if p:
        plots.append(p)

    # Save resampled summary
    if not resampled.empty:
        resampled_path = out / f"resampled_{value_col}_{resample_freq}.csv"
        resampled.to_csv(resampled_path, index=False)
        print(f"  ✓ Resampled summary saved → {resampled_path}")

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Rolling columns added : {rolling_cols + [cumulative_col]}")
    print(f"  Plots saved           : {len(plots)}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "date_col": date_col,
        "value_col": value_col,
        "rolling_windows": rolling_windows,
        "resample_freq": resample_freq,
        "rolling_columns_added": rolling_cols,
        "cumulative_column": cumulative_col,
        "trend": trend_result,
        "plots": plots,
    }

    return df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_timeseries_report(report: dict, output_path: str | Path) -> None:
    """Persist the time-series analysis report to JSON.

    Args:
        report: Report dict returned by run_timeseries_analysis().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Time-series report saved → {path}")
