from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from cardio_analysis.api import create_app
from cardio_analysis.schema import FEATURE_COLUMNS
from cardio_analysis.training import run_training


def test_health_and_prediction_endpoints(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "heart_failure_clinical_records_dataset.csv"
    model_path = tmp_path / "model.joblib"
    run_training(dataset, model_path)

    patient = (
        pd.read_csv(dataset).loc[0, list(FEATURE_COLUMNS)].to_dict()
    )
    client = TestClient(create_app(model_path))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_available"] is True

    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert abs(body["survival_probability"] + body["death_probability"] - 1) < 1e-6

