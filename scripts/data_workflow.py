from pathlib import Path

import pandas as pd

from data_ingestion import document_ingestion, ingest_data
from data_imputation import impute_missing_values, write_imputation_log
from data_merge_validation import (
    find_unmatched_keys,
    run_multi_source_merge,
    write_merge_report,
    write_unmatched_keys,
)
from data_validation import generate_validation_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
IMPUTATION_LOG = PROJECT_DIR / "output" / "imputation_report.json"
MERGE_REPORT = PROJECT_DIR / "output" / "merge_validation_report.json"
UNMATCHED_OUTPUT_DIR = PROJECT_DIR / "output"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]


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

        # Multi-source merge — build a unified view from the base table and any
        # supplementary sources.  Each merge is validated with before/after row
        # counts and unmatched key detection.
        #
        # Currently only one source is loaded (sample.csv).  Add additional
        # DataFrames to the sources list as new data files become available.
        # Example: product lookup, regional mapping, customer demographics.
        #
        # The first dict is always the base (left) table.  Subsequent dicts are
        # merged in order with their documented join type and business reason.
        sources = [
            {
                "df": imputed_data,
                "name": "transactions",
            },
            # Placeholder: uncomment and populate when a second source is ready
            # {
            #     "df": product_data,
            #     "key": "product_id",
            #     "how": "left",
            #     "name": "products",
            #     "reason": "Enrich transactions with product category and price tier",
            # },
        ]

        if len(sources) > 1:
            # Two or more sources — run the full validated multi-source merge
            merged_data, merge_report = run_multi_source_merge(sources)
            write_merge_report(merge_report, MERGE_REPORT)

            # Investigate unmatched keys between the first two sources
            left_unmatched, right_unmatched = find_unmatched_keys(
                sources[0]["df"],
                sources[1]["df"],
                key=sources[1]["key"],
            )
            write_unmatched_keys(
                left_unmatched,
                right_unmatched,
                output_dir=UNMATCHED_OUTPUT_DIR,
                left_name=sources[0].get("name", "left"),
                right_name=sources[1].get("name", "right"),
            )
        else:
            # Single source — no merge needed, pass through directly
            merged_data = imputed_data
            print("\n  Single source loaded — merge step skipped "
                  "(add sources to enable multi-source merge)\n")

        processed = process_data(merged_data)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)