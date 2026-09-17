"""Persistencia versionada do modelo e de seu pre-processamento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cardio_analysis.schema import FEATURE_COLUMNS, FEATURE_COLUMNS_PT

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ModelBundle:
    model: XGBClassifier
    scaler: StandardScaler
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    model_feature_columns: tuple[str, ...] = FEATURE_COLUMNS_PT
    artifact_version: int = ARTIFACT_VERSION

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        bundle: Any = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError("O arquivo nao contem um ModelBundle valido")
        if bundle.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                f"Versao de artefato incompativel: {bundle.artifact_version}"
            )
        return bundle
