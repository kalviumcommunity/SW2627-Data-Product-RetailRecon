from __future__ import annotations

from pathlib import Path

import pandas as pd


def _to_path(filepath: str | Path) -> Path:
    return filepath if isinstance(filepath, Path) else Path(filepath)


def ingest_csv(filepath: str | Path, delimiter: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Load CSV data with explicit delimiter and encoding parameters."""
    path = _to_path(filepath)

    encodings = [encoding, "latin-1", "iso-8859-1", "cp1252"]
    seen_encodings: set[str] = set()

    for current_encoding in encodings:
        if current_encoding in seen_encodings:
            continue
        seen_encodings.add(current_encoding)

        try:
            return pd.read_csv(path, delimiter=delimiter, encoding=current_encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not load CSV file with supported encodings: {path}")


def ingest_json(filepath: str | Path, is_nested: bool = False, encoding: str = "utf-8") -> pd.DataFrame:
    """Load JSON data and flatten nested records when requested."""
    path = _to_path(filepath)

    with path.open("r", encoding=encoding) as file_handle:
        raw_data = pd.read_json(file_handle)

    if is_nested:
        flattened = pd.json_normalize(raw_data.to_dict(orient="records"))
        print("✓ Flattened nested JSON")
        return flattened

    return raw_data


def ingest_data(
    filepath: str | Path,
    delimiter: str = ",",
    encoding: str = "utf-8",
    json_nested: bool = False,
) -> pd.DataFrame:
    """Dispatch ingestion based on file extension with explicit parameters."""
    path = _to_path(filepath)
    extension = path.suffix.lstrip(".").lower()

    if extension == "csv":
        return ingest_csv(path, delimiter=delimiter, encoding=encoding)
    if extension == "json":
        return ingest_json(path, is_nested=json_nested, encoding=encoding)

    raise ValueError(f"Unsupported file format: {extension}")


def document_ingestion(df: pd.DataFrame, source: str | Path) -> None:
    """Print a compact audit trail for the loaded dataset."""
    print(f"\nINGESTION REPORT: {source}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumn Types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))