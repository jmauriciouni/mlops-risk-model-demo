# src/simulate_production_data.py
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRODUCTION_DATA_DIR, PRODUCTION_DATA_PATH, RANDOM_STATE
from .generate_synthetic_data import generate_credit_risk_dataset


def apply_data_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simula data drift cambiando la distribución de channel e ingresos."""
    df = df.copy()

    # Más tráfico digital: subir proporción de 'web' y 'partner'
    new_channels = ["branch", "web", "partner", "call_center"]
    new_probs = [0.15, 0.45, 0.3, 0.10]  # antes branch 0.4, ahora mucho más web/partner
    df["channel"] = rng.choice(new_channels, size=len(df), p=new_probs)

    # Clientes más riesgosos: ingresos más bajos
    df["monthly_income"] = df["monthly_income"] * rng.uniform(0.6, 0.9, size=len(df))
    df["monthly_income"] = df["monthly_income"].clip(lower=800)

    return df


def apply_model_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simula model drift cambiando la relación X -> y (default_90d)."""
    df = df.copy()

    # Ignoramos la columna default_90d original y recalculamos con una regla distinta
    debt_to_income = df["debt_to_income"].values
    num_open_loans = df["num_open_loans"].values
    monthly_income = df["monthly_income"].values
    channel = df["channel"].values
    employment_type = df["employment_type"].values

    # Nuevo "mundo": ahora el canal 'partner' es mucho más riesgoso,
    # y el peso de debt_to_income se incrementa.
    score = (
        -1.0
        + 2.3 * debt_to_income
        + 0.8 * (num_open_loans > 3)
        + 0.7 * (channel == "partner")
        + 0.4 * (channel == "web")
        + 0.6 * (employment_type == "unemployed")
        + 0.3 * (monthly_income < 1800)
    )

    p_default_new = 1 / (1 + np.exp(-score))
    p_default_new = np.clip(p_default_new + rng.normal(0, 0.03, size=len(df)), 0.01, 0.98)
    df["default_90d"] = rng.binomial(1, p_default_new, size=len(df))

    return df


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE + 123)

    scenario = os.environ.get("DRIFT_SCENARIO", "both").lower()
    n_samples = int(os.environ.get("PRODUCTION_SAMPLES", 5000))

    print(f"Generando batch de producción con escenario: {scenario} (n={n_samples})")

    # Partimos de un dataset "normal"
    df = generate_credit_risk_dataset(n_samples=n_samples, random_state=RANDOM_STATE + 999)

    if scenario in ("data", "both"):
        df = apply_data_drift(df, rng)

    if scenario in ("model", "both"):
        df = apply_model_drift(df, rng)

    PRODUCTION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRODUCTION_DATA_PATH, index=False)
    print(f"Batch de producción guardado en: {PRODUCTION_DATA_PATH}")


if __name__ == "__main__":
    main()
