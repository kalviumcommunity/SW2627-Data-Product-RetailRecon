from pathlib import Path

import pandas as pd

from data_ingestion import document_ingestion, ingest_data
from data_type_enforcement import enforce_types, generate_type_report
from data_validation import generate_validation_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
TYPE_REPORT = PROJECT_DIR / "output" / "type_enforcement_report.json"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

# Declarative type map — update formats and columns to match your dataset.
# Supported types: 'datetime', 'currency', 'boolean'
TYPE_MAP = {
    "date":   {"type": "datetime", "fmt": "%Y-%m-%d"},
    "amount": {"type": "currency"},
}


def process_data(df: pd.DataFrame) -> pd.DataFrame:
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


def output_results(df: pd.DataFrame, output_path: Path) -> None:
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

        # Step 1: Validate
        validation_report = generate_validation_report(
            INPUT_FILE,
            EXPECTED_COLUMNS,
            report_path=VALIDATION_REPORT,
        )

        if not validation_report["passed"]:
            print("Validation failed. See output/intake_report.json for details.")
            raise SystemExit(1)

        # Step 2: Ingest
        data = ingest_data(INPUT_FILE, delimiter=",", encoding="utf-8", json_nested=False)
        document_ingestion(data, INPUT_FILE)

        # Step 3: Enforce types (before any analysis or cleaning)
        data_before = data.copy()
        data, conversion_logs = enforce_types(data, TYPE_MAP)
        generate_type_report(data_before, data, conversion_logs, report_path=TYPE_REPORT)

        # Step 4: Clean
        processed = process_data(data)

        # Step 5: Output
        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)
