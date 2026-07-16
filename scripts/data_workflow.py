from pathlib import Path

import pandas as pd

from data_behavioural_segmentation import run_behavioural_segmentation, write_segmentation_report
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
SEGMENTATION_REPORT = PROJECT_DIR / "output" / "behavioural_segmentation_report.json"
PLOTS_DIR = PROJECT_DIR / "output" / "plots"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

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