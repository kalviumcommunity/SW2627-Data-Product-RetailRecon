import pandas as pd

# File locations
INPUT_FILE = "../data/raw/sample.csv"
OUTPUT_FILE = "../output/processed.csv"


def ingest_data(filepath):
    """
    Read the CSV file.

    Input:
        CSV file path

    Returns:
        Pandas DataFrame
    """
    df = pd.read_csv(filepath)
    return df


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

        data = ingest_data(INPUT_FILE)

        processed = process_data(data)

        output_results(processed, OUTPUT_FILE)

        print("✓ Workflow completed successfully")

    except Exception as e:
        print("Error:", e)