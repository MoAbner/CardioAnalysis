from pathlib import Path

from cardio_analysis.validation import run_pipeline_test


def test_complete_pipeline() -> None:
    root = Path(__file__).resolve().parents[1]
    result = run_pipeline_test(root / "heart_failure_clinical_records_dataset.csv")
    assert result["status"] == "ok"
