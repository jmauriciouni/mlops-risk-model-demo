# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_risk.csv"

MODELS_DIR = BASE_DIR / "models"
LATEST_MODEL_PATH = MODELS_DIR / "model-latest.pkl"

RANDOM_STATE = 42
TARGET_COLUMN = "default_90d"

# --- Configuración MLflow ---
# Usamos un store local dentro del repo.
MLFLOW_DIR = BASE_DIR / "mlruns"
# MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}" # Linux WSL
# MLFLOW_TRACKING_URI = str(MLFLOW_DIR.resolve())
MLFLOW_TRACKING_URI = MLFLOW_DIR.as_uri() # Windows Powershell
MLFLOW_EXPERIMENT_NAME = "credit_risk_baseline"

# --- Métricas y datos de producción ---
METRICS_DIR = BASE_DIR / "metrics"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.json"
PRODUCTION_DATA_DIR = DATA_DIR / "production"
PRODUCTION_DATA_PATH = PRODUCTION_DATA_DIR / "production_batch.csv"
