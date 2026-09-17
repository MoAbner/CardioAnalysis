"""Contrato de dados compartilhado pelo treino, inferencia e API."""

from __future__ import annotations

from typing import Final

TARGET_COLUMN: Final = "DEATH_EVENT"
TARGET_COLUMN_PT: Final = "Evento de Óbito"

# A ordem e os nomes abaixo sao os mesmos do CSV/notebook. A ordem faz parte do
# contrato do scaler e do modelo e, portanto, nao deve ser alterada.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
)

COLUMN_TRANSLATIONS: Final[dict[str, str]] = {
    "age": "Idade",
    "anaemia": "Anemia",
    "creatinine_phosphokinase": "Creatina Fosfoquinase",
    "diabetes": "Diabetes",
    "ejection_fraction": "Fração de Ejeção",
    "high_blood_pressure": "Pressão Alta",
    "platelets": "Plaquetas",
    "serum_creatinine": "Creatinina Sérica",
    "serum_sodium": "Sódio Sérico",
    "sex": "Sexo",
    "smoking": "Fumante",
    "time": "Tempo",
    TARGET_COLUMN: TARGET_COLUMN_PT,
}

FEATURE_COLUMNS_PT: Final[tuple[str, ...]] = tuple(
    COLUMN_TRANSLATIONS[name] for name in FEATURE_COLUMNS
)

