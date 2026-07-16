from pathlib import Path

import pandas as pd

from data_deduplication import run_deduplication, write_deduplication_log, write_removed_records
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
DEDUP_LOG = PROJECT_DIR / "output" / "deduplication_report.json"
REMOVED_RECORDS = PROJECT_DIR / "output" / "removed_duplicates_audit.csv"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]


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

        # Detect and remove exact and near-duplicates with full audit trail
        deduped_data, dedup_report, removed_records = run_deduplication(
            imputed_data,
            key_columns=["customer_id", "date"],
            exact_keep="first",
            near_keep="first",
        )
        write_deduplication_log(dedup_report, DEDUP_LOG)
        write_removed_records(removed_records, REMOVED_RECORDS)

        processed = process_data(deduped_data)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)