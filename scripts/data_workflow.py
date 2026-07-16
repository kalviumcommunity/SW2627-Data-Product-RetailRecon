from pathlib import Path

import pandas as pd

from data_ingestion import document_ingestion, ingest_data
from data_imputation import impute_missing_values, write_imputation_log
from data_timeseries_rolling import run_timeseries_analysis, write_timeseries_report
from data_validation import generate_validation_report

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "sample.csv"
OUTPUT_FILE = PROJECT_DIR / "output" / "processed.csv"
VALIDATION_REPORT = PROJECT_DIR / "output" / "intake_report.json"
IMPUTATION_LOG = PROJECT_DIR / "output" / "imputation_report.json"
TIMESERIES_REPORT = PROJECT_DIR / "output" / "timeseries_rolling_report.json"
PLOTS_DIR = PROJECT_DIR / "output" / "plots"
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

        processed = process_data(imputed_data)

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

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)