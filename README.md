**English** | [Português](README.pt-BR.md)

# CardioAnalysis

Modular machine-learning application for the Heart Failure Clinical Records
dataset. It preserves the notebook's tabular XGBoost/Random Forest workflow and
exposes the final XGBoost model through a reusable Python class and FastAPI.

This is an educational project, not a clinically validated diagnostic tool.
Review the prospective use of the follow-up `time` feature before deployment.

## Setup and training

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e ".[test]"
cardio-train --data heart_failure_clinical_records_dataset.csv \
  --model artifacts/model.joblib --metrics artifacts/metrics.json
```

Run the end-to-end smoke test with `python scripts/test_pipeline.py` or `pytest`.

## Python integration

```python
from cardio_analysis import HeartFailurePredictor

predictor = HeartFailurePredictor.from_file("artifacts/model.joblib")
result = predictor.predict(patient_dict)
```

`patient_dict` must contain the 12 original CSV feature names. The result
contains the predicted class plus survival and death probabilities. Load one
predictor at application startup and reuse it across requests.

## HTTP API

```bash
uvicorn cardio_analysis.api:app --host 0.0.0.0 --port 8000
```

The API reads `artifacts/model.joblib` by default; override it with
`CARDIO_MODEL_PATH`. Open `/docs` for the schema, call `GET /health` for artifact
availability, and `POST /predict` for inference. See
[the Portuguese guide](README.pt-BR.md) for the full payload, project structure,
and continuous-retraining guidance.
