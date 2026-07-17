from pathlib import Path

import pandas as pd

from data_ingestion import document_ingestion, ingest_data
from data_type_enforcement import enforce_types, generate_type_report
from data_validation import generate_validation_report
from funnel_analysis import run_funnel_analysis

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
INPUT_FILE        = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE       = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
TYPE_REPORT       = PROJECT_DIR / "output" / "type_enforcement_report.json"
FUNNEL_REPORT     = PROJECT_DIR / "output" / "funnel_report.json"
PLOTS_DIR         = PROJECT_DIR / "output"

EXPECTED_COLUMNS = ["customer_id", "amount", "date"]

# ---------------------------------------------------------------------------
# Type enforcement config
# Supported types: 'datetime', 'currency', 'boolean'
# ---------------------------------------------------------------------------
TYPE_MAP = {
    "date":   {"type": "datetime", "fmt": "%Y-%m-%d"},
    "amount": {"type": "currency"},
}

# ---------------------------------------------------------------------------
# Funnel config
# Define ordered stages that map to binary columns in your dataset.
# Update stage names and column names to match your actual data.
# ---------------------------------------------------------------------------
FUNNEL_STAGES = {
    "Sign Up":        {"column": "signup_completed",  "value": 1},
    "Email Verified": {"column": "email_verified",    "value": 1},
    "Payment Added":  {"column": "payment_added",     "value": 1},
    "First Purchase": {"column": "first_purchase",    "value": 1},
}

# Average revenue a user generates after completing the funnel.
# Set to None to skip revenue impact calculations.
REVENUE_PER_USER: float | None = 150.0


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset.

    Removes duplicate rows and imputes missing numerical values with
    their column median.
    """
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def output_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed DataFrame to CSV."""
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

        # Step 3: Enforce types — always before analysis or cleaning
        data_before = data.copy()
        data, conversion_logs = enforce_types(data, TYPE_MAP)
        generate_type_report(data_before, data, conversion_logs, report_path=TYPE_REPORT)

        # Step 4: Clean
        processed = process_data(data)

        # Step 5: Funnel analysis — runs only if funnel columns are present
        funnel_cols = {spec["column"] for spec in FUNNEL_STAGES.values()}
        if funnel_cols.issubset(set(processed.columns)):
            run_funnel_analysis(
                processed,
                stage_definitions=FUNNEL_STAGES,
                revenue_per_user=REVENUE_PER_USER,
                output_dir=PLOTS_DIR,
                report_path=FUNNEL_REPORT,
                save_plots=True,
            )
        else:
            missing = funnel_cols - set(processed.columns)
            print(f"\n⚠ Funnel analysis skipped — columns not found: {missing}")
            print("  Update FUNNEL_STAGES in data_workflow.py to match your dataset columns.")

        # Step 6: Output
        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)
