from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from charset_normalizer import from_path


def _to_path(filepath: str | Path) -> Path:
    return filepath if isinstance(filepath, Path) else Path(filepath)


def validate_file_exists(filepath: str | Path) -> tuple[bool, str]:
    path = _to_path(filepath)

    if not path.exists():
        return False, f"File not found: {path}"
    if path.stat().st_size == 0:
        return False, "File is empty"
    return True, "File exists and has content"


def validate_file_format(filepath: str | Path, allowed: tuple[str, ...] = ("csv",)) -> tuple[bool, str]:
    path = _to_path(filepath)
    extension = path.suffix.lstrip(".").lower()

    if extension not in allowed:
        return False, f"Unsupported format: {extension}"
    return True, f"Format valid: {extension}"


def detect_encoding(filepath: str | Path) -> tuple[str, str]:
    path = _to_path(filepath)

    try:
        matches = from_path(path)
        best_match = matches.best()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return "unknown", f"Encoding detection failed: {exc}"

    if best_match is None:
        return "unknown", "Encoding could not be detected"

    encoding = best_match.encoding or "unknown"
    confidence = getattr(best_match, "percent_coherence", None)
    if confidence is None:
        confidence = getattr(best_match, "coherence", None)
        if isinstance(confidence, (int, float)) and confidence <= 1:
            confidence *= 100

    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence:.0f}%"
    else:
        confidence_text = "unknown confidence"

    return encoding, f"Detected: {encoding} ({confidence_text})"


def validate_schema(df: pd.DataFrame, expected_cols: list[str]) -> tuple[bool, str]:
    expected = set(expected_cols)
    actual = set(df.columns)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"Missing: {missing}")
        if extra:
            parts.append(f"Extra: {extra}")
        return False, " | ".join(parts)

    return True, "Schema valid"


def capture_stats(filepath: str | Path, df: pd.DataFrame) -> dict[str, float | int]:
    path = _to_path(filepath)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 4),
    }


def _write_report(report: dict, report_path: str | Path) -> None:
    path = _to_path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, default=str)


def generate_validation_report(
    filepath: str | Path,
    expected_cols: list[str],
    allowed_formats: tuple[str, ...] = ("csv",),
    report_path: str | Path = Path("../output/intake_report.json"),
) -> dict:
    path = _to_path(filepath)
    report = {
        "timestamp": datetime.now().isoformat(),
        "filepath": str(path),
        "checks": {},
        "statistics": {},
        "passed": False,
    }

    file_ok, file_message = validate_file_exists(path)
    report["checks"]["file_exists"] = file_message
    if not file_ok:
        _write_report(report, report_path)
        return report

    format_ok, format_message = validate_file_format(path, allowed_formats)
    report["checks"]["format"] = format_message
    if not format_ok:
        _write_report(report, report_path)
        return report

    encoding, encoding_message = detect_encoding(path)
    report["checks"]["encoding"] = encoding_message

    try:
        df = pd.read_csv(path, encoding=encoding if encoding != "unknown" else "utf-8")
    except Exception as exc:
        report["checks"]["read"] = f"Could not read file: {exc}"
        _write_report(report, report_path)
        return report

    schema_ok, schema_message = validate_schema(df, expected_cols)
    report["checks"]["schema"] = schema_message
    if not schema_ok:
        _write_report(report, report_path)
        return report

    report["statistics"] = capture_stats(path, df)
    report["passed"] = True

    _write_report(report, report_path)
    return report