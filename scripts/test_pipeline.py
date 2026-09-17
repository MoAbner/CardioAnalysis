"""Smoke test executavel: CSV -> treino -> artefato -> predicao."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardio_analysis.validation import run_pipeline_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida o pipeline completo")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("heart_failure_clinical_records_dataset.csv"),
    )
    args = parser.parse_args()
    print(json.dumps(run_pipeline_test(args.data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
