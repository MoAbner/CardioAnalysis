"""Validacao ponta a ponta reutilizavel do pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from cardio_analysis.inference import HeartFailurePredictor
from cardio_analysis.schema import FEATURE_COLUMNS
from cardio_analysis.training import run_training


def run_pipeline_test(dataset_path: Path) -> dict[str, object]:
    """Executa CSV -> treino -> persistencia -> carga -> predicao."""

    with tempfile.TemporaryDirectory(prefix="cardio_pipeline_") as temp_dir:
        model_path = Path(temp_dir) / "model.joblib"
        metrics_path = Path(temp_dir) / "metrics.json"
        training = run_training(dataset_path, model_path, metrics_path)

        first_patient = (
            pd.read_csv(dataset_path).loc[0, list(FEATURE_COLUMNS)].to_dict()
        )
        predictor = HeartFailurePredictor.from_file(model_path)
        prediction = predictor.predict(first_patient)

        direct_frame = predictor._to_model_frame(first_patient)
        direct_scaled = training.bundle.scaler.transform(direct_frame)
        expected = int(training.bundle.model.predict(direct_scaled)[0])
        assert prediction["prediction"] == expected
        assert model_path.is_file()
        assert metrics_path.is_file()
        assert abs(
            prediction["survival_probability"]
            + prediction["death_probability"]
            - 1.0
        ) < 1e-6

        return {
            "status": "ok",
            "metrics": training.metrics,
            "sample_prediction": prediction,
        }

