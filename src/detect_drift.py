# src/detect_drift.py
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from .config import (
    PROCESSED_DATA_DIR,
    PRODUCTION_DATA_PATH,
    BASELINE_METRICS_PATH,
    TARGET_COLUMN,
    LATEST_MODEL_PATH,
)
from .model_utils import split_features_target
import joblib


def load_train_reference() -> pd.DataFrame:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"No se encontró {train_path}. Ejecuta primero src.data_prep.")
    df_train = pd.read_csv(train_path, parse_dates=["application_date"])
    return df_train


def load_production_batch() -> pd.DataFrame:
    if not PRODUCTION_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró batch de producción en {PRODUCTION_DATA_PATH}. Ejecuta src.simulate_production_data primero."
        )
    df_prod = pd.read_csv(PRODUCTION_DATA_PATH, parse_dates=["application_date"])
    return df_prod


def load_baseline_metrics() -> dict:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {BASELINE_METRICS_PATH}. Entrena el modelo (src.train) para generar métricas baseline."
        )
    with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_data_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame) -> bool:
    """Heurística simple de data drift: cambios en 'channel' y 'monthly_income'."""

    # Distribución de channel
    def channel_dist(df):
        return df["channel"].value_counts(normalize=True).reindex(
            ["branch", "web", "partner", "call_center"], fill_value=0.0
        )

    ch_train = channel_dist(train_df)
    ch_prod = channel_dist(prod_df)

    tvd_channel = 0.5 * np.abs(ch_train - ch_prod).sum()  # Total Variation Distance

    # Promedio de ingreso
    mean_income_train = train_df["monthly_income"].mean()
    mean_income_prod = prod_df["monthly_income"].mean()
    rel_change_income = abs(mean_income_prod - mean_income_train) / mean_income_train

    print("=== DATA DRIFT CHECK ===")
    print(f"Distribución channel (train):\n{ch_train}")
    print(f"Distribución channel (prod):\n{ch_prod}")
    print(f"TVD channel: {tvd_channel:.3f}")
    print(f"Mean income train: {mean_income_train:.2f}")
    print(f"Mean income prod : {mean_income_prod:.2f}")
    print(f"Relative change income: {rel_change_income:.3f}")

    # Umbrales simples
    tvd_threshold = 0.25       # si cambia más de 25% la distribución de canales
    income_threshold = 0.20    # si el promedio cambia más de 20%

    data_drift = (tvd_channel > tvd_threshold) or (rel_change_income > income_threshold)
    print(f"DATA_DRIFT = {data_drift} (tvd>{tvd_threshold} or rel_change>{income_threshold})")

    return data_drift


def compute_model_drift(prod_df: pd.DataFrame, baseline_metrics: dict) -> bool:
    """Model drift: degradación de métricas vs baseline."""

    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró modelo en {LATEST_MODEL_PATH}. Entrena primero el modelo (src.train)."
        )

    model = joblib.load(LATEST_MODEL_PATH)
    X_prod, y_prod = split_features_target(prod_df, TARGET_COLUMN)

    y_proba = model.predict_proba(X_prod)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc_prod = roc_auc_score(y_prod, y_proba)
    f1_prod = f1_score(y_prod, y_pred)

    print("=== MODEL DRIFT CHECK ===")
    print(f"Baseline AUC: {baseline_metrics['auc_valid']:.4f}")
    print(f"Prod AUC    : {auc_prod:.4f}")
    print(f"Baseline F1 : {baseline_metrics['f1_valid']:.4f}")
    print(f"Prod F1     : {f1_prod:.4f}")

    # Umbral: caída de 0.05 en AUC o F1
    auc_drop = baseline_metrics["auc_valid"] - auc_prod
    f1_drop = baseline_metrics["f1_valid"] - f1_prod

    print(f"AUC drop: {auc_drop:.4f}")
    print(f"F1  drop: {f1_drop:.4f}")

    drift_auc = auc_drop > 0.05 # umbral de 0.05
    drift_f1 = f1_drop > 0.05 # umbral de 0.05

    model_drift = drift_auc or drift_f1
    print(f"MODEL_DRIFT = {model_drift} (auc_drop>0.05 or f1_drop>0.05)")

    return model_drift


def main() -> None:
    print("=== DETECCIÓN DE DRIFT (DATA + MODEL) ===")
    train_df = load_train_reference()
    prod_df = load_production_batch()
    baseline_metrics = load_baseline_metrics()

    data_drift = compute_data_drift(train_df, prod_df)
    model_drift = compute_model_drift(prod_df, baseline_metrics)

    retrain = data_drift or model_drift
    print("=========================================")
    print(f"Resultado final → data_drift={data_drift}, model_drift={model_drift}, retrain={retrain}")

    # Integración con GitHub Actions: exportar salida para condicionarlo en el workflow
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"retrain={str(retrain).lower()}\n")
            f.write(f"data_drift={str(data_drift).lower()}\n")
            f.write(f"model_drift={str(model_drift).lower()}\n")


if __name__ == "__main__":
    main()
