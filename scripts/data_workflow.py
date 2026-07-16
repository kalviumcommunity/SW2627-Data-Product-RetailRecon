from pathlib import Path

import pandas as pd

from data_groupby_aggregation import run_groupby_analysis, write_groupby_report
from data_ingestion import document_ingestion, ingest_data
from data_imputation import impute_missing_values, write_imputation_log
from data_validation import generate_validation_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
IMPUTATION_LOG = PROJECT_DIR / "output" / "imputation_report.json"
GROUPBY_REPORT = PROJECT_DIR / "output" / "groupby_segment_report.json"
PLOTS_DIR = PROJECT_DIR / "output" / "plots"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

# ---------------------------------------------------------------------------
# Segment analysis config — each dict defines one groupby analysis.
# Never report dataset-wide statistics; always segment first.
# ---------------------------------------------------------------------------
SEGMENT_CONFIGS = [
    # Single-dimension: revenue and transaction metrics by region
    {
        "group_by": "region",
        "agg_config": {
            "amount": ["sum", "mean", "count"],
        },
        "rename_map": {
            "amount_sum": "total_revenue",
            "amount_mean": "avg_transaction",
            "amount_count": "transaction_count",
        },
        "rank_by": "total_revenue",
        "rank_ascending": False,         # highest revenue first
        "plot_metric": "total_revenue",
        "transform_col": "amount",       # broadcast regional mean back to rows
        "transform_func": "mean",
        # Pivot: region × customer_id to see per-customer contribution per region
        "pivot": {
            "row_key": "region",
            "col_key": "customer_id",
            "value_col": "amount",
            "aggfunc": "sum",
        },
    },
    # Single-dimension: average transaction and count by customer
    {
        "group_by": "customer_id",
        "agg_config": {
            "amount": ["sum", "mean", "count"],
        },
        "rename_map": {
            "amount_sum": "customer_total",
            "amount_mean": "customer_avg",
            "amount_count": "customer_transactions",
        },
        "rank_by": "customer_total",
        "rank_ascending": False,
        "plot_metric": "customer_total",
    },
]


def process_data(df):
    """
    Clean the dataset.

    Input:
        Raw DataFrame

    Returns:
        Clean DataFrame
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing numerical values with median
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

        processed = process_data(imputed_data)

        # GroupBy segment analysis — never report dataset-wide stats, always segment
        processed, groupby_report = run_groupby_analysis(
            processed,
            segment_configs=SEGMENT_CONFIGS,
            output_dir=PLOTS_DIR,
        )
        write_groupby_report(groupby_report, GROUPBY_REPORT)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)