from pathlib import Path

import pandas as pd

from data_consistency_validation import (
    run_consistency_validation,
    write_validation_failures,
    write_validation_report,
)
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
CONSISTENCY_REPORT = PROJECT_DIR / "output" / "consistency_validation_report.json"
VALIDATION_FAILURES = PROJECT_DIR / "output" / "validation_failures.csv"
EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

# ---------------------------------------------------------------------------
# Validation rules — one dict per rule, applied in order.
# Extend this list as the schema grows; no changes needed elsewhere.
# ---------------------------------------------------------------------------
VALIDATION_RULES = [
    # Null constraint — customer_id must never be empty
    {
        "name": "valid_customer_id",
        "type": "not_null",
        "column": "customer_id",
        "description": "customer_id must not be null (critical identifier)",
    },
    # Range check — amount must be non-negative
    {
        "name": "valid_amount_range",
        "type": "range",
        "column": "amount",
        "min_value": 0,
        "description": "Transaction amount must be >= 0 (negative amounts are invalid)",
    },
    # Range check — amount must be below a reasonable ceiling
    {
        "name": "valid_amount_ceiling",
        "type": "range",
        "column": "amount",
        "max_value": 1_000_000,
        "description": "Transaction amount must be <= 1,000,000 (catches data entry errors)",
    },
    # Format check — customer_id must follow pattern C + digits (e.g. C001)
    {
        "name": "valid_customer_id_format",
        "type": "format",
        "column": "customer_id",
        "pattern": r"^C\d+$",
        "description": "customer_id must match pattern C<digits> (e.g. C001, C123)",
    },
    # Business rule — date must not be in the future
    {
        "name": "valid_date_not_future",
        "type": "business",
        "rule_func": lambda df: (
            pd.to_datetime(df["date"], errors="coerce") <= pd.Timestamp.now()
            if "date" in df.columns else pd.Series(True, index=df.index)
        ),
        "description": "Transaction date must not be in the future",
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

        # Apply consistency validation rules — only clean records proceed
        clean_data, failures, consistency_report = run_consistency_validation(
            imputed_data,
            VALIDATION_RULES,
        )
        write_validation_report(consistency_report, CONSISTENCY_REPORT)
        write_validation_failures(failures, VALIDATION_FAILURES)

        processed = process_data(clean_data)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)