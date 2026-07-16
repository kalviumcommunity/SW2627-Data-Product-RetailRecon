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
# Feature engineering config — extend as the dataset grows.
# Ratios are created first so binning can reference them.
# ---------------------------------------------------------------------------

# Ratio features: normalise amount by days as a spend-rate proxy
RATIO_CONFIG = [
    {
        "name": "amount_per_day",
        "numerator": "amount",
        "denominator": lambda df: (
            (pd.Timestamp.now() - pd.to_datetime(df["date"], errors="coerce")).dt.days
        ),
        "description": "Transaction amount normalised by days since date (spend rate proxy)",
    },
]

# Binned features: segment customers by spend tier
BIN_CONFIG = [
    {
        "name": "amount_tier",
        "column": "amount",
        "strategy": "cut",
        "bins": [float("-inf"), 0, 100, 300, float("inf")],
        "labels": ["negative", "low", "medium", "high"],
        "description": "Transaction amount tier: negative / low (<100) / medium / high (>300)",
    },
    {
        "name": "amount_quantile_tier",
        "column": "amount",
        "strategy": "qcut",
        "q": 4,
        "labels": ["tier_1", "tier_2", "tier_3", "tier_4"],
        "description": "Amount quartile tier — equal-frequency segments",
    },
]

# Composite scores: RFM-style score on amount (monetary component only for now)
SCORE_CONFIG = [
    {
        "name": "amount_score",
        "components": [
            {
                "source": "amount",
                "q": 5,
                "labels": [1, 2, 3, 4, 5],
                "temp_col": "monetary_score",
            },
        ],
        "keep_components": False,
        "description": "Monetary score 1-5 (quintile rank of transaction amount)",
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


        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)