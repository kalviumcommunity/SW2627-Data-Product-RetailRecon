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
# Step 1: Compute correlation matrices
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute a correlation matrix for all numeric columns.

    Two methods are available — choose based on your data:

    Pearson  — measures LINEAR relationship between two continuous variables.
               Assumes approximately normal distribution.
               Sensitive to outliers (one extreme value can inflate r).
               Use when the relationship is expected to be linear.

    Spearman — measures MONOTONIC relationship using rank order.
               Makes no distribution assumption.
               Robust to outliers and non-linear monotonic trends.
               Use for ordinal data or when linearity is uncertain.

    r ranges from -1 to +1:
      r >  0.7 : strong positive  — one high, other tends high
      r < -0.7 : strong negative  — one high, other tends low
      |r| < 0.3: weak relationship — little predictive signal

    Args:
        df: DataFrame containing numeric columns to correlate.
        method: 'pearson' or 'spearman'.
        min_periods: Minimum observations required per pair (default 1).

    Returns:
        Square correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(method=method, min_periods=min_periods)


# ---------------------------------------------------------------------------
# Step 2: Find strong pairs
# ---------------------------------------------------------------------------

def find_strong_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Flatten the correlation matrix and return pairs above the threshold.

    Flattening (unstack) converts the matrix into a Series of (var1, var2) pairs
    so strong relationships can be ranked and inspected without reading a grid.

    r > threshold  : strong positive — features move together, possible redundancy
    r < -threshold : strong negative — features move inversely

    Args:
        corr_matrix: Output of compute_correlations().
        threshold: Absolute r value above which a pair is considered strong.
        exclude_self: Drop pairs where var1 == var2 (r = 1.0 always).

    Returns:
        DataFrame with columns [var1, var2, correlation] sorted by |r| descending.
    """
    flat = corr_matrix.unstack().reset_index()
    flat.columns = ["var1", "var2", "correlation"]

    if exclude_self:
        flat = flat[flat["var1"] != flat["var2"]]

    # Keep only one direction of each pair (A<->B, not B<->A duplicate)
    flat = flat[flat["var1"] < flat["var2"]]

    strong = flat[flat["correlation"].abs() >= threshold].copy()
    strong = strong.sort_values("correlation", key=abs, ascending=False)
    strong = strong.reset_index(drop=True)

    return strong


# ---------------------------------------------------------------------------
# Step 3: Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> str:
    """Visualise the full correlation matrix as a colour-coded heatmap.

    Heatmap colour intensity shows correlation strength:
      Red (warm)  = strong positive — both variables rise together
      Blue (cool) = strong negative — one rises, the other falls
      White       = no relationship

    Annotated values let you read the exact r without estimating from colour.

    Args:
        corr_matrix: Square correlation matrix DataFrame.
        method: Method label ('pearson' or 'spearman') used in title/filename.
        output_dir: Directory to save the PNG.

    Returns:
        Path string of the saved PNG file.
    """
    if corr_matrix.empty:
        print(f"  ⚠  Heatmap skipped — correlation matrix is empty")
        return ""

    n = len(corr_matrix)
    fig_size = max(8, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(
        f"Correlation Matrix ({method.capitalize()})\n"
        f"Red = positive | Blue = negative | Intensity = strength",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()

    output_path = output_dir / f"correlation_heatmap_{method}.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Heatmap saved → {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Step 4: Scatter plot for strong pairs
# ---------------------------------------------------------------------------

def plot_strong_pair_scatters(
    df: pd.DataFrame,
    strong_pairs: pd.DataFrame,
    output_dir: Path,
    max_pairs: int = 6,
) -> list[str]:
    """Plot scatter plots for the top strong correlation pairs.

    Scatter plots confirm whether the relationship is truly linear
    (Pearson assumption), reveals outlier influence, and shows whether
    the pattern supports a business hypothesis.

    Args:
        df: Input DataFrame with the raw column values.
        strong_pairs: Output of find_strong_correlations().
        output_dir: Directory to save PNGs.
        max_pairs: Maximum number of pairs to plot.

    Returns:
        List of saved PNG path strings.
    """
    if strong_pairs.empty:
        print(f"  ℹ  No strong pairs to plot (none exceed threshold)")
        return []

    pairs_to_plot = strong_pairs.head(max_pairs)
    plots: list[str] = []

    for _, row in pairs_to_plot.iterrows():
        v1, v2, r = row["var1"], row["var2"], row["correlation"]

        if v1 not in df.columns or v2 not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[v1], df[v2], alpha=0.6, edgecolors="white",
                   linewidths=0.3, color="steelblue", s=40)
        ax.set_xlabel(v1)
        ax.set_ylabel(v2)
        ax.set_title(f"{v1}  ←→  {v2}\nr = {r:.2f}", fontsize=11)

        # Trend line
        valid = df[[v1, v2]].dropna()
        if len(valid) >= 2:
            z = pd.Series(valid[v2].values).values
            x = pd.Series(valid[v1].values).values
            m, b = pd.Series(z).cov(pd.Series(x)) / pd.Series(x).var(), 0
            m = (valid[v2].cov(valid[v1])) / (valid[v1].var())
            b = valid[v2].mean() - m * valid[v1].mean()
            x_line = pd.Series([valid[v1].min(), valid[v1].max()])
            ax.plot(x_line, m * x_line + b, color="red",
                    linewidth=1.5, linestyle="--", label="trend")
            ax.legend(fontsize=8)

        plt.tight_layout()
        safe_name = f"{v1}_vs_{v2}".replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"scatter_{safe_name}.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(output_path))
        print(f"  ✓ Scatter saved → {output_path}  (r={r:.2f})")

    return plots


# ---------------------------------------------------------------------------
# Step 5: Causation warning generator
# ---------------------------------------------------------------------------

def generate_causation_warnings(strong_pairs: pd.DataFrame) -> list[dict]:
    """Generate correlation-≠-causation reminders for every strong pair.

    Correlation says "they move together".
    Causation says "one makes the other happen".
    Three possibilities for every correlated pair A and B:
      1. A causes B
      2. B causes A
      3. C causes both A and B (confounding variable)

    This function attaches a structured warning to every strong pair so
    that anyone reading the report is reminded to reason about direction
    before drawing conclusions.

    Args:
        strong_pairs: Output of find_strong_correlations().

    Returns:
        List of warning dicts, one per pair.
    """
    warnings: list[dict] = []
    for _, row in strong_pairs.iterrows():
        v1, v2, r = row["var1"], row["var2"], row["correlation"]
        direction = "positive" if r > 0 else "negative"

        warning = {
            "pair": f"{v1} <-> {v2}",
            "r": round(float(r), 4),
            "direction": direction,
            "correlation_statement": (
                f"'{v1}' and '{v2}' have a {direction} correlation (r={r:.2f}): "
                f"they tend to {'both increase' if r > 0 else 'move in opposite directions'} together."
            ),
            "causation_warning": (
                f"⚠  Correlation ≠ Causation. Three possibilities: "
                f"(1) {v1} causes {v2}, "
                f"(2) {v2} causes {v1}, "
                f"(3) a third confounding variable drives both. "
                f"Investigate direction before acting on this relationship."
            ),
        }
        warnings.append(warning)

    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_correlation_analysis(
    df: pd.DataFrame,
    output_dir: str | Path,
    methods: list[str] | None = None,
    strong_threshold: float = 0.7,
) -> dict:
    """Run the full correlation analysis pipeline for all numeric columns.

    For each method (Pearson + Spearman):
      1. Compute correlation matrix
      2. Find pairs with |r| >= threshold
      3. Plot colour-coded heatmap
      4. Plot scatter plots for strong pairs
      5. Generate correlation-≠-causation warnings for every strong pair

    Running both Pearson and Spearman together reveals whether relationships
    are linear (Pearson r ≈ Spearman r) or monotonic-but-nonlinear
    (Spearman r >> Pearson r).

    Args:
        df: Processed DataFrame with numeric columns to analyse.
        output_dir: Directory for saving plots and the JSON report.
        methods: Correlation methods to run (default ['pearson', 'spearman']).
        strong_threshold: |r| cutoff for labelling a pair "strong".

    Returns:
        Full correlation analysis report dict.
    """
    out = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = methods or ["pearson", "spearman"]
    method_reports: list[dict] = []
    all_plots: list[str] = []

    print("\n── Correlation & Relationship Analysis ────────────────────────")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    print(f"\n  Numeric columns: {numeric_cols}")
    print(f"  Strong threshold: |r| >= {strong_threshold}")

    for method in methods:
        print(f"\n  ── {method.capitalize()} correlation ─────────────────────")

        # Step 1: Compute
        corr_matrix = compute_correlations(df, method=method)
        if corr_matrix.empty:
            print(f"  ⚠  No numeric columns — skipped")
            continue

        # Step 2: Strong pairs
        strong_pairs = find_strong_correlations(corr_matrix, threshold=strong_threshold)
        print(f"  Strong pairs (|r|>={strong_threshold}): {len(strong_pairs)}")
        for _, row in strong_pairs.iterrows():
            print(f"    {row['var1']} <-> {row['var2']}: {row['correlation']:.2f}")

        # Step 3: Heatmap
        heatmap_path = plot_correlation_heatmap(corr_matrix, method, out)
        if heatmap_path:
            all_plots.append(heatmap_path)

        # Step 4: Scatter plots for strong pairs
        scatter_paths = plot_strong_pair_scatters(df, strong_pairs, out)
        all_plots.extend(scatter_paths)

        # Step 5: Causation warnings
        warnings = generate_causation_warnings(strong_pairs)
        if warnings:
            print(f"\n  Causation warnings ({len(warnings)}):")
            for w in warnings:
                print(f"    {w['causation_warning']}")

        method_reports.append({
            "method": method,
            "numeric_columns": numeric_cols,
            "strong_pairs": strong_pairs.to_dict(orient="records"),
            "heatmap": heatmap_path,
            "scatter_plots": scatter_paths,
            "causation_warnings": warnings,
        })

    print(f"\n── Summary ───────────────────────────────────────────────────")
    print(f"  Methods run  : {methods}")
    print(f"  Plots saved  : {len(all_plots)}")
    print("──────────────────────────────────────────────────────────────\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "strong_threshold": strong_threshold,
        "numeric_columns": numeric_cols,
        "plots": all_plots,
        "analysis": method_reports,
    }

    return report


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_correlation_report(report: dict, output_path: str | Path) -> None:
    """Persist the correlation analysis report to JSON.

    Args:
        report: Report dict returned by run_correlation_analysis().
        output_path: Destination path for the JSON file.
    """
    path = output_path if isinstance(output_path, Path) else Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"✓ Correlation analysis report saved → {path}")
