"""Inferencia independente de notebook, pronta para uso por outros servicos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cardio_analysis.artifacts import ModelBundle
from cardio_analysis.schema import COLUMN_TRANSLATIONS


class HeartFailurePredictor:
    """Carrega uma vez o modelo e aplica o mesmo scaler usado no treinamento."""

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle

    @classmethod
    def from_file(cls, model_path: str | Path) -> "HeartFailurePredictor":
        return cls(ModelBundle.load(model_path))

    def _to_model_frame(self, raw_data: Mapping[str, Any]) -> pd.DataFrame:
        raw_columns = set(self._bundle.feature_columns)
        model_columns = set(self._bundle.model_feature_columns)
        received = set(raw_data)

        if received == raw_columns:
            translated = {
                COLUMN_TRANSLATIONS[column]: raw_data[column]
                for column in self._bundle.feature_columns
            }
        elif received == model_columns:
            translated = {
                column: raw_data[column]
                for column in self._bundle.model_feature_columns
            }
        else:
            missing = sorted(raw_columns - received)
            extra = sorted(received - raw_columns)
            raise ValueError(
                "Entrada deve conter exatamente as 12 features em ingles (CSV) "
                "ou em portugues (modelo). "
                f"Ausentes: {missing}; extras: {extra}"
            )

        frame = pd.DataFrame([translated], columns=self._bundle.model_feature_columns)
        try:
            numeric = frame.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Todas as features devem ser numericas") from error
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("As features nao podem conter NaN ou infinito")
        return numeric

    def predict(self, raw_data: Mapping[str, Any]) -> dict[str, float | int]:
        """Retorna classe e probabilidades para um unico paciente."""

        frame = self._to_model_frame(raw_data)
        scaled = self._bundle.scaler.transform(frame)
        probabilities = self._bundle.model.predict_proba(scaled)[0]
        prediction = self._bundle.model.predict(scaled)[0]
        return {
            "prediction": int(prediction),
            "survival_probability": float(probabilities[0]),
            "death_probability": float(probabilities[1]),
        }

    def predict_many(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, float | int]]:
        return [self.predict(record) for record in records]

    def analyze(self, raw_data: Mapping[str, Any]) -> dict[str, Any]:
        """Versao estruturada da analise de risco interativa do notebook."""

        frame = self._to_model_frame(raw_data)
        patient = frame.iloc[0].to_dict()
        current = self.predict(patient)
        alerts: list[str] = []

        if patient["Fração de Ejeção"] < 40:
            alerts.append("Fração de ejeção abaixo de 40%")
        if patient["Creatinina Sérica"] > 1.3:
            alerts.append("Creatinina sérica acima de 1,3 mg/dL")
        if patient["Sódio Sérico"] < 135:
            alerts.append("Sódio sérico abaixo de 135 mEq/L")
        if patient["Fumante"] == 1:
            alerts.append("Paciente fumante")
        if patient["Tempo"] < 50:
            alerts.append("Tempo de acompanhamento abaixo de 50 dias")

        # Valores usados pela simulacao de "cenario ideal" no notebook. O valor
        # 250.000 e mantido como 250.0, conforme a semantica literal do Python.
        ideal = patient.copy()
        ideal.update(
            {
                "Fumante": 0,
                "Creatinina Sérica": 1.0,
                "Fração de Ejeção": 50,
                "Sódio Sérico": 140,
                "Tempo": 100,
                "Plaquetas": 250.000,
                "Pressão Alta": 0,
                "Creatina Fosfoquinase": 150,
                "Anemia": 0,
            }
        )
        ideal_result = self.predict(ideal)
        reduction = current["death_probability"] - ideal_result["death_probability"]
        return {
            **current,
            "alerts": alerts,
            "ideal_scenario_death_probability": ideal_result["death_probability"],
            "absolute_risk_reduction": float(reduction),
        }

