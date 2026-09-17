"""Pipeline de treinamento executavel sem Jupyter."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

from cardio_analysis.artifacts import ModelBundle
from cardio_analysis.data import PreparedData, load_dataset, prepare_training_data


@dataclass(frozen=True)
class TrainingResult:
    bundle: ModelBundle
    data: PreparedData
    metrics: dict[str, Any]
    shallow_xgboost: XGBClassifier
    random_forest: RandomForestClassifier


def train_pipeline(dataset_path: str | Path) -> TrainingResult:
    """Treina os modelos do notebook e seleciona seu XGBoost final."""

    dataframe = load_dataset(dataset_path)
    data = prepare_training_data(dataframe)

    # Modelos comparativos da celula 5, mantidos com os mesmos parametros.
    shallow_xgboost = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    shallow_xgboost.fit(data.x_train_scaled, data.y_train)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(data.x_train_scaled, data.y_train)

    # Este segundo XGBoost sobrescreve o anterior no notebook e e o modelo usado
    # pela matriz de confusao e pela inferencia interativa.
    production_model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
    production_model.fit(data.x_train_scaled, data.y_train)

    predictions = production_model.predict(data.x_test_scaled)
    probabilities = production_model.predict_proba(data.x_test_scaled)[:, 1]
    metrics: dict[str, Any] = {
        "train_samples": len(data.x_train),
        "test_samples": len(data.x_test),
        "confusion_matrix": confusion_matrix(data.y_test, predictions).tolist(),
        "f1_score": float(f1_score(data.y_test, predictions)),
        "roc_auc": float(roc_auc_score(data.y_test, probabilities)),
        "classification_report": classification_report(
            data.y_test, predictions, output_dict=True
        ),
    }
    return TrainingResult(
        bundle=ModelBundle(model=production_model, scaler=data.scaler),
        data=data,
        metrics=metrics,
        shallow_xgboost=shallow_xgboost,
        random_forest=random_forest,
    )


def run_training(
    dataset_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
) -> TrainingResult:
    result = train_pipeline(dataset_path)
    result.bundle.save(model_path)
    if metrics_path is not None:
        output = Path(metrics_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo CardioAnalysis")
    parser.add_argument("--data", required=True, help="CSV de treinamento")
    parser.add_argument(
        "--model", default="artifacts/model.joblib", help="Artefato de saida"
    )
    parser.add_argument(
        "--metrics", default="artifacts/metrics.json", help="Metricas JSON"
    )
    args = parser.parse_args()
    result = run_training(args.data, args.model, args.metrics)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
