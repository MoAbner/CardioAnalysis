"""Carregamento e pre-processamento reproduzivel dos dados clinicos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cardio_analysis.schema import (
    COLUMN_TRANSLATIONS,
    FEATURE_COLUMNS_PT,
    TARGET_COLUMN_PT,
)


@dataclass(frozen=True)
class PreparedData:
    """Dados resultantes do mesmo split e scaling usados no notebook."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_train_scaled: np.ndarray
    x_test_scaled: np.ndarray
    scaler: StandardScaler


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Le o CSV, remove espacos dos cabecalhos e traduz as colunas."""

    dataframe = pd.read_csv(path)
    dataframe.columns = dataframe.columns.str.strip()
    required = set(COLUMN_TRANSLATIONS)
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")
    return dataframe.rename(columns=COLUMN_TRANSLATIONS)


def prepare_training_data(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
) -> PreparedData:
    """Executa exatamente o split 70/30 e StandardScaler do notebook."""

    x = dataframe.drop(TARGET_COLUMN_PT, axis=1)
    y = dataframe[TARGET_COLUMN_PT]
    if tuple(x.columns) != FEATURE_COLUMNS_PT:
        raise ValueError(
            "Ordem ou nomes das features inesperados. "
            f"Esperado: {list(FEATURE_COLUMNS_PT)}; recebido: {list(x.columns)}"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_train_scaled=x_train_scaled,
        x_test_scaled=x_test_scaled,
        scaler=scaler,
    )
