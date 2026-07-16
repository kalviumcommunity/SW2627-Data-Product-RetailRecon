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
# Step 1: Single-dimension segment aggregation (.agg)
# ---------------------------------------------------------------------------

def compute_segment_metrics(
    df: pd.DataFrame,
    group_by: str | list[str],
    agg_config: dict[str, list[str]],
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics per segment using .agg() — the core split-apply-combine.

    Never report dataset-wide statistics — always segment first.
    Average churn of 5% is meaningless until you see enterprise at 1%
    and SMB at 12%.

    .agg() returns one row per group with the requested aggregations
    collapsed to a single value per group — the "combine" step.

    Args:
        df: Input DataFrame.
        group_by: Column name(s) to group by (segment key).
        agg_config: Dict mapping column → list of aggregation functions.
                    e.g. {'amount': ['sum', 'mean', 'count']}
        rename_map: Optional dict renaming output columns for readability.
                    e.g. {'amount_sum': 'total_revenue'}

    Returns:
        Flat segment metrics DataFrame, one row per segment.
    """
    keys = [group_by] if isinstance(group_by, str) else list(group_by)

    result = df.groupby(keys).agg(agg_config)

    # Flatten multi-level column names: ('amount', 'sum') → 'amount_sum'
    result.columns = ["_".join(col).strip() for col in result.columns]
    result = result.reset_index()

    if rename_map:
        result = result.rename(columns=rename_map)

    return result


# ---------------------------------------------------------------------------
# Step 2: .transform — broadcast group metric back to original row shape
# ---------------------------------------------------------------------------

def add_group_benchmarks(
    df: pd.DataFrame,
    group_by: str | list[str],
    column: str,
    func: str = "mean",
) -> pd.DataFrame:
    """Broadcast a group-level metric back to every row using .transform().

    .agg() collapses groups to one row.
    .transform() returns a value for every original row, aligned to its group.
    This lets you compare each row to its segment benchmark:
        row_amount vs the mean amount for that customer's region.

    Args:
        df: Input DataFrame.
        group_by: Column(s) defining the group.
        column: Numeric column to aggregate per group.
        func: Aggregation function ('mean', 'median', 'sum', 'count').

    Returns:
        DataFrame with a new column {column}_{func}_by_{group_by} appended.
    """
    keys = [group_by] if isinstance(group_by, str) else list(group_by)
    key_label = "_".join(keys)
    out_col = f"{column}_{func}_by_{key_label}"

    working_df = df.copy()
    # .transform returns same shape as original — broadcasts group result per row
    working_df[out_col] = df.groupby(keys)[column].transform(func)

    print(f"  ✓ [{out_col}] — group {func} of '{column}' broadcast to every row")
    return working_df


# ---------------------------------------------------------------------------
# Step 3: Multi-level groupby and pivot table
# ---------------------------------------------------------------------------

def compute_multidimensional_aggregation(
    df: pd.DataFrame,
    row_key: str,
    col_key: str,
    value_col: str,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    """Build a pivot table across two segment dimensions simultaneously.

    Groups by two keys at once — e.g. customer_type × product — and
    surfaces the interaction between segments:
        "Which customer type spends most on which product category?"
        "Where does revenue concentration exist across two dimensions?"

    pd.pivot_table handles missing combinations cleanly (fill_value=0).

    Args:
        df: Input DataFrame.
        row_key: Column whose unique values become pivot rows.
        col_key: Column whose unique values become pivot columns.
        value_col: Numeric column to aggregate.
        aggfunc: Aggregation function — 'sum', 'mean', 'count'.

    Returns:
        Pivot DataFrame with row_key as index and col_key values as columns.
    """
    if row_key not in df.columns or col_key not in df.columns:
        print(f"  ⚠  Pivot skipped — '{row_key}' or '{col_key}' not found")
        return pd.DataFrame()

    pivot = pd.pivot_table(
        df,
        values=value_col,
        index=row_key,
        columns=col_key,
        aggfunc=aggfunc,
        fill_value=0,
    )
    pivot = pivot.reset_index()

    print(f"  ✓ Pivot table: {row_key} × {col_key} → {value_col} ({aggfunc})")
    print(f"    Shape: {pivot.shape[0]} rows × {pivot.shape[1]} columns")

    return pivot


# ---------------------------------------------------------------------------
# Step 4: Rank segments and surface actionable insights
# ---------------------------------------------------------------------------

def rank_and_surface_insights(
    segment_df: pd.DataFrame,
    segment_col: str,
    rank_by: str,
    ascending: bool = False,
    top_n: int = 3,
    insight_template: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank segments by a key metric and generate actionable insight strings.

    Ranking transforms a table of numbers into a prioritised action list:
    top performers are identified for replication, bottom performers for
    intervention.  Plain-English insight strings make the output
    immediately usable in reports without further interpretation.

    Args:
        segment_df: Output of compute_segment_metrics() or similar.
        segment_col: Column identifying the segment label.
        rank_by: Metric column to rank on.
        ascending: True = lowest first (e.g. worst churn), False = highest first.
        top_n: Number of top and bottom segments to highlight.
        insight_template: Optional f-string template for custom insights.
                          Available variables: segment, rank, value.
                          Default: "Segment '{segment}' ranks #{rank} with
                          {rank_by}={value:.2f}"

    Returns:
        (ranked DataFrame with rank column, list of actionable insight strings)
    """
    if rank_by not in segment_df.columns:
        return segment_df, [f"⚠ Cannot rank — '{rank_by}' not found"]

    ranked = segment_df.copy()
    ranked["rank"] = ranked[rank_by].rank(ascending=ascending, method="min").astype(int)
    ranked = ranked.sort_values("rank").reset_index(drop=True)

    insights: list[str] = []

    print(f"\n  Segment ranking by '{rank_by}' ({'asc' if ascending else 'desc'}):")

    for _, row in ranked.iterrows():
        seg = row[segment_col]
        rank = row["rank"]
        value = row[rank_by]

        if insight_template:
            insight = insight_template.format(segment=seg, rank=rank, value=value)
        else:
            insight = (
                f"Segment '{seg}' ranks #{rank} — "
                f"{rank_by} = {value:.2f}"
            )

        if rank <= top_n:
            insight = f"🔝 TOP    {insight}"
        elif rank > len(ranked) - top_n:
            insight = f"⚠️  BOTTOM {insight}"

        insights.append(insight)
        print(f"    {insight}")

    return ranked, insights


# ---------------------------------------------------------------------------
# Step 5: Segment bar chart
# ---------------------------------------------------------------------------

def plot_segment_bar(
    segment_df: pd.DataFrame,
    segment_col: str,
    value_col: str,
    output_dir: Path,
    title: str | None = None,
    colour: str = "steelblue",
) -> str:
    """Bar chart showing a metric across segments — ranked for easy reading.

    Sorted bar charts make the best and worst performers visible at a glance
    without needing to read a table row by row.

    Args:
        segment_df: Aggregated segment DataFrame.
        segment_col: Column with segment labels (x-axis).
        value_col: Metric column (bar height).
        output_dir: Directory to save the PNG.
        title: Chart title (auto-generated if None).
        colour: Bar fill colour.

    Returns:
        Path string of the saved PNG file.
    """
    if segment_col not in segment_df.columns or value_col not in segment_df.columns:
        print(f"  ⚠  Bar chart skipped — column missing")
        return ""

    sorted_df = segment_df.sort_values(value_col, ascending=False)

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_df) * 1.2), 5))
    bars = ax.bar(
        sorted_df[segment_col].astype(str),
        sorted_df[value_col],
        color=colour,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate bar tops
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            f"{h:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel(segment_col, fontsize=10)
    ax.set_ylabel(value_col, fontsize=10)
    ax.set_title(title or f"{value_col} by {segment_col}", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    safe_name = f"segment_{value_col}_by_{segment_col}".replace(" ", "_")
    output_path = output_dir / f"{safe_name}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Segment bar chart saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_groupby_analysis(
    df: pd.DataFrame,
    segment_configs: list[dict],
    output_dir: str | Path,
) -> dict:
    """Run the full groupby segment analysis pipeline.

    Each entry in segment_configs defines one analysis:
        {
            "group_by"     : "region",          # segment key (str or list)
            "agg_config"   : {"amount": ["sum", "mean", "count"]},
            "rename_map"   : {"amount_sum": "total_revenue",
                              "amount_mean": "avg_transaction",
                              "amount_count": "transaction_count"},
            "rank_by"      : "total_revenue",   # column to rank segments on
            "rank_ascending": False,            # highest first
            "plot_metric"  : "total_revenue",   # column for bar chart
            "transform_col": "amount",          # .transform broadcast column
            "transform_func": "mean",           # function for .transform
            "pivot"        : {                  # optional pivot table
                "row_key"  : "region",
                "col_key"  : "customer_id",
                "value_col": "amount",
                "aggfunc"  : "sum",
            },
        }

    Args:
        df: Processed DataFrame ready for segmentation.
        segment_configs: List of segment analysis config dicts.
        output_dir: Directory for saving plots, pivot CSVs, and JSON report.

    Returns:
        Full groupby analysis report dict.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    analysis_results: list[dict] = []
    all_plots: list[str] = []

    print("\n── GroupBy Aggregation & Segment Insights ─────────────────────")

    for i, cfg in enumerate(segment_configs):
        group_by = cfg["group_by"]
        agg_config = cfg.get("agg_config", {})
        rename_map = cfg.get("rename_map")
        rank_by = cfg.get("rank_by")
        rank_ascending = cfg.get("rank_ascending", False)
        plot_metric = cfg.get("plot_metric")
        transform_col = cfg.get("transform_col")
        transform_func = cfg.get("transform_func", "mean")
        pivot_cfg = cfg.get("pivot")

        group_label = group_by if isinstance(group_by, str) else " × ".join(group_by)
        print(f"\n  [{i+1}] Group by: {group_label}")

        result: dict = {
            "group_by": group_by,
            "segment_metrics": None,
            "ranked_insights": [],
            "pivot": None,
            "plots": [],
        }

        # Step 1: .agg — collapse to one row per segment
        if agg_config:
            segment_metrics = compute_segment_metrics(
                df, group_by, agg_config, rename_map=rename_map
            )
            print(f"    Segments found: {len(segment_metrics)}")
            print(segment_metrics.to_string(index=False))
            result["segment_metrics"] = segment_metrics.to_dict(orient="records")

            # Step 2: Rank + insights
            if rank_by and rank_by in segment_metrics.columns:
                seg_col = group_by if isinstance(group_by, str) else group_by[0]
                ranked, insights = rank_and_surface_insights(
                    segment_metrics,
                    segment_col=seg_col,
                    rank_by=rank_by,
                    ascending=rank_ascending,
                )
                result["ranked_insights"] = insights

            # Step 3: Bar chart
            if plot_metric and plot_metric in segment_metrics.columns:
                seg_col = group_by if isinstance(group_by, str) else group_by[0]
                plot_path = plot_segment_bar(
                    segment_metrics,
                    segment_col=seg_col,
                    value_col=plot_metric,
                    output_dir=out,
                )
                if plot_path:
                    all_plots.append(plot_path)
                    result["plots"].append(plot_path)

        # Step 2b: .transform — broadcast group metric back to rows
        if transform_col and transform_col in df.columns:
            df = add_group_benchmarks(df, group_by, transform_col, func=transform_func)

        # Step 4: Pivot table
        if pivot_cfg:
            pivot_df = compute_multidimensional_aggregation(
                df,
                row_key=pivot_cfg["row_key"],
                col_key=pivot_cfg["col_key"],
                value_col=pivot_cfg["value_col"],
                aggfunc=pivot_cfg.get("aggfunc", "sum"),
            )
            if not pivot_df.empty:
                pivot_path = out / f"pivot_{pivot_cfg['row_key']}_x_{pivot_cfg['col_key']}.csv"
                pivot_df.to_csv(pivot_path, index=False)
                print(f"  ✓ Pivot table saved → {pivot_path}")
                result["pivot"] = str(pivot_path)

        analysis_results.append(result)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Segment configs run : {len(segment_configs)}")
    print(f"  Plots saved         : {len(all_plots)}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "segment_configs_run": len(segment_configs),
        "plots": all_plots,
        "analysis": analysis_results,
    }

    return df, report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_groupby_report(report: dict, output_path: str | Path) -> None:
    """Persist the groupby analysis report to JSON.

    Args:
        report: Report dict returned by run_groupby_analysis().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ GroupBy analysis report saved → {path}")
