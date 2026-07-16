from pathlib import Path

import pandas as pd

from data_ingestion import document_ingestion, ingest_data
from data_imputation import impute_missing_values, write_imputation_log
from data_string_cleaning import clean_string_columns, write_string_cleaning_log
from data_validation import generate_validation_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
IMPUTATION_LOG = PROJECT_DIR / "output" / "imputation_report.json"
STRING_CLEANING_LOG = PROJECT_DIR / "output" / "string_cleaning_report.json"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

# ---------------------------------------------------------------------------
# Column-level string cleaning config
# Each key is a column name; value is cleaning options for clean_text_column().
# Extend this dict as the schema grows — no code changes needed elsewhere.
# ---------------------------------------------------------------------------
STRING_CLEANING_CONFIG = {
    # customer_id: strip whitespace only — preserve original casing for IDs
    "customer_id": {"strip": True, "lowercase": False},
    # region: strip + lowercase so "North", " NORTH ", "north" all unify
    "region": {"strip": True, "lowercase": True},
    # notes: strip + lowercase + remove special characters for safe text export
    "notes": {"strip": True, "lowercase": True, "remove_special": True},
}


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

        # Normalise string columns before analysis
        # Strips whitespace, unifies casing, removes special characters
        cleaned_data, string_report = clean_string_columns(imputed_data, STRING_CLEANING_CONFIG)
        write_string_cleaning_log(string_report, STRING_CLEANING_LOG)

        processed = process_data(cleaned_data)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)