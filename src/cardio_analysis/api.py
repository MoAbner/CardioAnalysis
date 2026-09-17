"""API HTTP fina sobre a classe de inferencia."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from cardio_analysis.inference import HeartFailurePredictor


class PatientInput(BaseModel):
    """Payload externo usando os nomes originais do dataset."""

    model_config = ConfigDict(extra="forbid")

    age: float
    anaemia: Literal[0, 1]
    creatinine_phosphokinase: float
    diabetes: Literal[0, 1]
    ejection_fraction: float
    high_blood_pressure: Literal[0, 1]
    platelets: float
    serum_creatinine: float
    serum_sodium: float
    sex: Literal[0, 1]
    smoking: Literal[0, 1]
    time: float


class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    survival_probability: float = Field(ge=0, le=1)
    death_probability: float = Field(ge=0, le=1)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="CardioAnalysis API",
        version="1.0.0",
        description="Modelo educacional; nao e uma ferramenta de diagnostico clinico.",
    )
    selected_path = Path(
        model_path or os.getenv("CARDIO_MODEL_PATH", "artifacts/model.joblib")
    )
    loaded_predictor: HeartFailurePredictor | None = None

    def get_predictor() -> HeartFailurePredictor:
        nonlocal loaded_predictor
        if loaded_predictor is None:
            try:
                loaded_predictor = HeartFailurePredictor.from_file(selected_path)
            except (FileNotFoundError, OSError, TypeError, ValueError) as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Modelo indisponivel em '{selected_path}': {error}",
                ) from error
        return loaded_predictor

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok" if selected_path.is_file() else "model_missing",
            "model_path": str(selected_path),
            "model_available": selected_path.is_file(),
        }

    @app.post("/predict", response_model=PredictionOutput)
    def predict(
        patient: PatientInput,
        predictor: HeartFailurePredictor = Depends(get_predictor),
    ) -> dict[str, float | int]:
        return predictor.predict(patient.model_dump())

    return app


app = create_app()
