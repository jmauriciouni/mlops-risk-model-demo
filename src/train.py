# src/train.py
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

import mlflow
import mlflow.sklearn
import json
from .config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    LATEST_MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    METRICS_DIR,
    BASELINE_METRICS_PATH,
)
from .model_utils import split_features_target, build_model_pipeline


def main() -> None:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "valid.csv"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            "No se encontraron train.csv / valid.csv. "
            "Ejecuta primero: dvc repro (o python -m src.data_prep)."
        )

    # 1) Cargar datos
    train_df = pd.read_csv(train_path, parse_dates=["application_date"])
    valid_df = pd.read_csv(valid_path, parse_dates=["application_date"])

    X_train, y_train = split_features_target(train_df, TARGET_COLUMN)
    X_valid, y_valid = split_features_target(valid_df, TARGET_COLUMN)

    # 2) Construir pipeline de modelo
    model = build_model_pipeline(random_state=RANDOM_STATE)

    # 3) Configurar MLflow (tracking local)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="logreg_baseline"):

        # 3.1) Log de parámetros
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_state": RANDOM_STATE,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
            }
        )

        # 4) Entrenar
        model.fit(X_train, y_train)

        # 5) Evaluar
        y_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_valid, y_proba)
        f1 = f1_score(y_valid, y_pred)

        print(f"AUC valid: {auc:.4f}")
        print(f"F1  valid: {f1:.4f}")

        # --- Guardar métricas baseline para detectar model drift ---
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_metrics = {
            "auc_valid": float(auc),
            "f1_valid": float(f1),
        }
        with open(BASELINE_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)
        print(f"Métricas baseline guardadas en: {BASELINE_METRICS_PATH}")

        # 5.1) Log de métricas
        mlflow.log_metrics(
            {
                "auc_valid": auc,
                "f1_valid": f1,
            }
        )

        # 5.2) Log del modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,  # puedes usar un Model Registry si quieres
        )

        # 6) Guardar el modelo "oficial" para la API
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, LATEST_MODEL_PATH)
        print(f"Modelo guardado en: {LATEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
