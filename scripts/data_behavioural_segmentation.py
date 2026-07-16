from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Non-interactive backend — safe for scripts running without a display
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Step 1: Summary metrics table per segment
# ---------------------------------------------------------------------------

def compute_segment_summary(
    df: pd.DataFrame,
    segment_col: str,
    agg_config: dict[str, list[str]],
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute multi-metric summary table grouped by segment.

    The core pattern from HRS2.32:
        df.groupby('customer_type').agg({
            'lifetime_value': 'mean',
            'churn': 'mean',
            'customer_id': 'count'
        })

    Aggregate reporting hides everything.  This table makes segment
    differences visible side-by-side — enterprise at 1% churn vs
    SMB at 12% churn in the same row.

    Args:
        df: Input DataFrame.
        segment_col: Column whose unique values define the segments.
        agg_config: Dict mapping column → list of aggregation functions.
                    e.g. {'amount': ['mean', 'sum', 'count']}
        rename_map: Optional flat column rename after flattening.
                    e.g. {'amount_mean': 'avg_transaction'}

    Returns:
        Flat segment summary DataFrame, one row per segment.
    """
    if segment_col not in df.columns:
        print(f"  ⚠  Segment column '{segment_col}' not found — skipped")
        return pd.DataFrame()

    result = df.groupby(segment_col).agg(agg_config)
    # Flatten multi-level column index: ('amount', 'mean') → 'amount_mean'
    result.columns = ["_".join(col).strip() for col in result.columns]
    result = result.reset_index()

    if rename_map:
        result = result.rename(columns=rename_map)

    return result


# ---------------------------------------------------------------------------
# Step 2: Identify top and bottom performers
# ---------------------------------------------------------------------------

def identify_top_bottom(
    segment_df: pd.DataFrame,
    segment_col: str,
    metric_cols: list[str],
) -> dict:
    """Find the top and bottom segment for each metric.

    Surfaces the extreme performers that require attention:
    highest value = candidate for investment/replication,
    lowest value  = candidate for intervention/investigation.

    Args:
        segment_df: Output of compute_segment_summary().
        segment_col: Column holding segment labels.
        metric_cols: Numeric metric columns to rank on.

    Returns:
        Dict mapping metric → {top: segment_label, bottom: segment_label}.
    """
    insights: dict = {}

    print(f"\n  Top / Bottom performers:")
    for col in metric_cols:
        if col not in segment_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(segment_df[col]):
            continue

        top_idx = segment_df[col].idxmax()
        bottom_idx = segment_df[col].idxmin()
        top_seg = segment_df.loc[top_idx, segment_col]
        bottom_seg = segment_df.loc[bottom_idx, segment_col]
        top_val = segment_df.loc[top_idx, col]
        bottom_val = segment_df.loc[bottom_idx, col]

        insights[col] = {
            "top": {"segment": str(top_seg), "value": round(float(top_val), 4)},
            "bottom": {"segment": str(bottom_seg), "value": round(float(bottom_val), 4)},
        }

        print(f"    [{col}]  highest={top_seg} ({top_val:.2f})  "
              f"lowest={bottom_seg} ({bottom_val:.2f})")

    return insights


# ---------------------------------------------------------------------------
# Step 3: Heatmap — colour intensity shows metric values across segments
# ---------------------------------------------------------------------------

def plot_segment_heatmap(
    segment_df: pd.DataFrame,
    segment_col: str,
    metric_cols: list[str],
    output_dir: Path,
    title: str = "Segment Comparison Heatmap",
) -> str:
    """Normalised heatmap comparing multiple metrics across all segments.

    Colour intensity reveals patterns instantly that tables hide:
    which segments are high on all metrics, which are low on one but
    high on another, where strategies need to differ.

    Each column is min-max normalised so metrics with different scales
    (e.g. count vs percentage) are comparable by colour.

    Args:
        segment_df: Flat segment summary DataFrame.
        segment_col: Column holding segment labels (used as row index).
        metric_cols: Numeric columns to include in the heatmap.
        output_dir: Directory to save the PNG.
        title: Chart title.

    Returns:
        Path string of the saved PNG.
    """
    numeric_cols = [
        c for c in metric_cols
        if c in segment_df.columns and pd.api.types.is_numeric_dtype(segment_df[c])
    ]

    if not numeric_cols:
        print(f"  ⚠  Heatmap skipped — no numeric metric columns found")
        return ""

    heat_df = segment_df.set_index(segment_col)[numeric_cols].copy()

    # Min-max normalise each column so colour is comparable across metrics
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min()).replace(0, 1)

    n_rows, n_cols = heat_norm.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.6), max(4, n_rows * 0.9)))

    sns.heatmap(
        heat_norm,
        annot=heat_df.round(2),   # show raw values, not normalised
        fmt="g",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalised value (0=min, 1=max)"},
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(segment_col)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    output_path = output_dir / f"heatmap_{segment_col}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Segment heatmap saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Step 4: Box plots — distribution within each segment
# ---------------------------------------------------------------------------

def plot_segment_boxplots(
    df: pd.DataFrame,
    segment_col: str,
    value_cols: list[str],
    output_dir: Path,
) -> list[str]:
    """Box plots showing distribution within each segment for each metric.

    A summary table shows averages.  A box plot shows the spread:
    where are the outliers, how wide is the IQR, do segments overlap?
    Two segments can have the same mean but very different distributions —
    the box plot reveals this; a bar chart does not.

    Args:
        df: Input DataFrame with raw row-level data.
        segment_col: Column defining the segments (x-axis).
        value_cols: Numeric columns to plot (one chart per column).
        output_dir: Directory to save PNGs.

    Returns:
        List of saved PNG path strings.
    """
    if segment_col not in df.columns:
        return []

    plots: list[str] = []

    for col in value_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        fig, ax = plt.subplots(figsize=(max(7, df[segment_col].nunique() * 1.5), 5))

        # Group data per segment for boxplot
        segments = sorted(df[segment_col].dropna().unique())
        data_per_segment = [
            df.loc[df[segment_col] == seg, col].dropna().values
            for seg in segments
        ]

        bp = ax.boxplot(
            data_per_segment,
            labels=[str(s) for s in segments],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )

        # Colour each box differently
        colours = plt.cm.tab10.colors
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_xlabel(segment_col)
        ax.set_ylabel(col)
        ax.set_title(f"{col} distribution by {segment_col}\n"
                     f"(box=IQR, whiskers=1.5×IQR, dots=outliers)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        output_path = output_dir / f"boxplot_{col}_by_{segment_col}.png"
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(output_path))
        print(f"  ✓ Box plot saved → {output_path}")

    return plots


# ---------------------------------------------------------------------------
# Step 5: Grouped bar chart — side-by-side metric comparison
# ---------------------------------------------------------------------------

def plot_segment_grouped_bar(
    segment_df: pd.DataFrame,
    segment_col: str,
    metric_cols: list[str],
    output_dir: Path,
    title: str = "Segment Metric Comparison",
) -> str:
    """Grouped bar chart comparing multiple metrics across segments.

    Each cluster of bars is one segment; each bar within the cluster
    is one metric.  Makes relative differences across segments immediately
    visible for stakeholder presentations.

    Args:
        segment_df: Flat segment summary DataFrame.
        segment_col: Column holding segment labels.
        metric_cols: Numeric metric columns to plot.
        output_dir: Directory to save the PNG.
        title: Chart title.

    Returns:
        Path string of the saved PNG.
    """
    numeric_cols = [
        c for c in metric_cols
        if c in segment_df.columns and pd.api.types.is_numeric_dtype(segment_df[c])
    ]

    if not numeric_cols or segment_col not in segment_df.columns:
        return ""

    plot_df = segment_df.set_index(segment_col)[numeric_cols]

    ax = plot_df.plot(
        kind="bar",
        figsize=(max(9, len(plot_df) * 1.5), 5),
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(segment_col)
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path = output_dir / f"grouped_bar_{segment_col}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Grouped bar chart saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Step 6: Generate actionable business insights
# ---------------------------------------------------------------------------

def generate_segment_insights(
    segment_df: pd.DataFrame,
    segment_col: str,
    top_bottom: dict,
) -> list[str]:
    """Produce plain-English actionable insight strings per segment.

    Translates metric numbers into strategy language:
    "Segment X has the highest revenue but the lowest transaction count —
    these are high-value, low-frequency customers. Prioritise retention."

    Args:
        segment_df: Flat segment summary DataFrame.
        segment_col: Column holding segment labels.
        top_bottom: Output of identify_top_bottom().

    Returns:
        List of actionable insight strings.
    """
    insights: list[str] = []

    print(f"\n  Actionable insights:")
    for metric, extremes in top_bottom.items():
        top_seg = extremes["top"]["segment"]
        top_val = extremes["top"]["value"]
        bottom_seg = extremes["bottom"]["segment"]
        bottom_val = extremes["bottom"]["value"]

        insight = (
            f"'{top_seg}' leads on {metric} ({top_val:.2f}) — "
            f"investigate what drives this and replicate. "
            f"'{bottom_seg}' is lowest ({bottom_val:.2f}) — "
            f"this segment requires targeted intervention."
        )
        insights.append(insight)
        print(f"    → {insight}")

    return insights


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_behavioural_segmentation(
    df: pd.DataFrame,
    segment_configs: list[dict],
    output_dir: str | Path,
) -> dict:
    """Run the full behavioural segmentation and comparison pipeline.

    Each entry in segment_configs defines one segmentation analysis:
        {
            "segment_col"  : "region",
            "agg_config"   : {"amount": ["mean", "sum", "count"]},
            "rename_map"   : {"amount_mean": "avg_transaction",
                              "amount_sum":  "total_revenue",
                              "amount_count": "transactions"},
            "rank_metrics" : ["total_revenue", "avg_transaction"],
            "box_cols"     : ["amount"],   # raw distribution plots
        }

    Three comparison perspectives per segment:
      1. Summary table   — metrics side-by-side (easy to scan)
      2. Heatmap         — colour intensity across all metrics × segments
      3. Box plots       — distribution within each segment
      4. Grouped bar     — side-by-side bar chart per metric

    Args:
        df: Processed DataFrame with the segment and value columns.
        segment_configs: List of segmentation config dicts.
        output_dir: Directory for saving plots and the JSON report.

    Returns:
        Full behavioural segmentation report dict.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_plots: list[str] = []

    print("\n── Behavioural Analysis & User Segmentation ───────────────────")

    for i, cfg in enumerate(segment_configs):
        segment_col = cfg["segment_col"]
        agg_config = cfg.get("agg_config", {})
        rename_map = cfg.get("rename_map")
        rank_metrics = cfg.get("rank_metrics", [])
        box_cols = cfg.get("box_cols", [])

        print(f"\n  [{i+1}] Segment by: '{segment_col}'")

        result: dict = {
            "segment_col": segment_col,
            "summary": [],
            "top_bottom": {},
            "insights": [],
            "plots": [],
        }

        # Step 1: Summary table
        summary_df = compute_segment_summary(df, segment_col, agg_config, rename_map)
        if summary_df.empty:
            all_results.append(result)
            continue

        print(f"\n  Summary table ({len(summary_df)} segments):")
        print(summary_df.to_string(index=False))
        result["summary"] = summary_df.to_dict(orient="records")

        # Resolve actual metric column names after rename
        numeric_metric_cols = [
            c for c in summary_df.columns
            if c != segment_col and pd.api.types.is_numeric_dtype(summary_df[c])
        ]

        # Step 2: Top/bottom performers
        effective_rank = [c for c in rank_metrics if c in summary_df.columns] \
                         or numeric_metric_cols
        if effective_rank:
            top_bottom = identify_top_bottom(summary_df, segment_col, effective_rank)
            result["top_bottom"] = top_bottom

            # Step 6: Actionable insights
            insights = generate_segment_insights(summary_df, segment_col, top_bottom)
            result["insights"] = insights

        # Step 3: Heatmap
        p = plot_segment_heatmap(
            summary_df, segment_col, numeric_metric_cols, out,
            title=f"Segment Heatmap — {segment_col}",
        )
        if p:
            all_plots.append(p)
            result["plots"].append(p)

        # Step 4: Box plots — raw distribution per segment
        if box_cols:
            bp_paths = plot_segment_boxplots(df, segment_col, box_cols, out)
            all_plots.extend(bp_paths)
            result["plots"].extend(bp_paths)

        # Step 5: Grouped bar chart
        p = plot_segment_grouped_bar(
            summary_df, segment_col, numeric_metric_cols, out,
            title=f"Segment Metric Comparison — {segment_col}",
        )
        if p:
            all_plots.append(p)
            result["plots"].append(p)

        all_results.append(result)

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Segment configs run : {len(segment_configs)}")
    print(f"  Plots saved         : {len(all_plots)}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "segment_configs_run": len(segment_configs),
        "plots": all_plots,
        "analysis": all_results,
    }

    return report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_segmentation_report(report: dict, output_path: str | Path) -> None:
    """Persist the behavioural segmentation report to JSON.

    Args:
        report: Report dict returned by run_behavioural_segmentation().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Behavioural segmentation report saved → {path}")
