from pathlib import Path

import pandas as pd


from data_ingestion import document_ingestion, ingest_data
from data_imputation import impute_missing_values, write_imputation_log


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
IMPUTATION_LOG = PROJECT_DIR / "output" / "imputation_report.json"

# ---------------------------------------------------------------------------
# Behavioural segmentation config.
# Each dict defines one segment analysis: what to group by, what to measure,
# and which metrics to rank and visualise.
# ---------------------------------------------------------------------------
SEGMENT_CONFIGS = [
    {
        "segment_col": "region",
        "agg_config": {
            "amount": ["mean", "sum", "count"],
        },
        "rename_map": {
            "amount_mean":  "avg_transaction",
            "amount_sum":   "total_revenue",
            "amount_count": "transaction_count",
        },
        "rank_metrics": ["total_revenue", "avg_transaction"],
        "box_cols": ["amount"],
    },
    {
        "segment_col": "customer_id",
        "agg_config": {
            "amount": ["sum", "mean", "count"],
        },
        "rename_map": {
            "amount_sum":   "customer_revenue",
            "amount_mean":  "customer_avg_spend",
            "amount_count": "customer_transactions",
        },
        "rank_metrics": ["customer_revenue"],
        "box_cols": [],
    },
]


def process_data(df):
    """
    Clean the dataset.

    Input:
        Raw DataFrame (post-deduplication)

    Returns:
        Clean DataFrame
    """
    # Fill missing numerical values with median
    # Note: duplicate removal is handled explicitly by run_deduplication()
    # before this step so every removal is logged for audit purposes.
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def output_results(df, output_path):
    """
    Save processed data.

    Input:
        Processed DataFrame

    Returns:
        None
    """

    df.to_csv(output_path, index=False)

    print("✓ Data successfully processed")
    print(f"✓ Rows processed: {len(df)}")
    print(f"✓ Output saved to {output_path}")


if __name__ == "__main__":
    try:
        print("Starting workflow...")

        validation_report = generate_validation_report(
            INPUT_FILE,
            EXPECTED_COLUMNS,
            report_path=VALIDATION_REPORT,
        )

        if not validation_report["passed"]:
            print("Validation failed. See output/intake_report.json for details.")
            raise SystemExit(1)

        data = ingest_data(INPUT_FILE, delimiter=",", encoding="utf-8", json_nested=False)
        document_ingestion(data, INPUT_FILE)

        imputed_data, imputation_report = impute_missing_values(
            data,
            critical_columns=["customer_id"],
            time_series_columns=["date"],
        )
        write_imputation_log(imputation_report, IMPUTATION_LOG)



        # Analyse distributions after cleaning — always visualise before reporting
        distribution_report = run_distribution_analysis(
            processed,
            columns=["amount"],
            output_dir=PLOTS_DIR,
            segment_comparisons=[
                {"column": "amount", "segment_column": "region"},
            ],
        )
        write_distribution_report(distribution_report, DISTRIBUTION_REPORT)

        # Correlation analysis — always analyse relationships before modelling
        # Pearson: linear relationships | Spearman: monotonic, robust to outliers
        correlation_report = run_correlation_analysis(
            processed,
            output_dir=PLOTS_DIR,
            methods=["pearson", "spearman"],
            strong_threshold=0.7,
        )
        write_correlation_report(correlation_report, CORRELATION_REPORT)

        # GroupBy segment analysis — never report dataset-wide stats, always segment
        processed, groupby_report = run_groupby_analysis(
            processed,
            segment_configs=SEGMENT_CONFIGS,
            output_dir=PLOTS_DIR,
        )
        write_groupby_report(groupby_report, GROUPBY_REPORT)

        # Time-series trend and rolling metrics analysis
        # Rolling averages smooth noise → cumulative sum → resample →
        # period-over-period change → trend identification
        processed, ts_report = run_timeseries_analysis(
            processed,
            date_col="date",
            value_col="amount",
            output_dir=PLOTS_DIR,
            rolling_windows=[7, 30],
            resample_freq="W",
            trend_lookback=3,
        )
        write_timeseries_report(ts_report, TIMESERIES_REPORT)

        # Behavioural segmentation — compare metrics across user/operational segments
        segmentation_report = run_behavioural_segmentation(
            processed,
            segment_configs=SEGMENT_CONFIGS,
            output_dir=PLOTS_DIR,
        )
        write_segmentation_report(segmentation_report, SEGMENTATION_REPORT)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)