import numpy as np
import pandas as pd

def generate_credit_risk_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # 1) customer_id
    customer_id = [f"C{str(i).zfill(6)}" for i in range(1, n_samples + 1)]

    # 2) application_date (distribuimos fechas en ~3 años)
    start_date = np.datetime64("2022-01-01")
    end_date = np.datetime64("2024-12-31")
    n_days = (end_date - start_date).astype(int)
    application_date = start_date + rng.integers(0, n_days, size=n_samples).astype("timedelta64[D]")

    # 3) age
    age = rng.integers(18, 76, size=n_samples)

    # 4) gender
    gender = rng.choice(["M", "F"], size=n_samples, p=[0.5, 0.5])

    # 5) marital_status
    marital_status = rng.choice(
        ["single", "married", "divorced", "widowed"],
        size=n_samples,
        p=[0.45, 0.4, 0.1, 0.05],
    )

    # 6) dependents
    dependents = rng.integers(0, 6, size=n_samples)

    # 7) monthly_income – asimétrica (más prob de ingresos bajos)
    monthly_income = rng.gamma(shape=2.0, scale=1500.0, size=n_samples) + 800
    monthly_income = np.clip(monthly_income, 800, 15000)

    # 8) employment_type
    employment_type = rng.choice(
        ["permanent", "contract", "self_employed", "unemployed"],
        size=n_samples,
        p=[0.55, 0.2, 0.2, 0.05],
    )

    # 9) employment_months – más corto para desempleados / contract
    base_emp_months = rng.integers(0, 361, size=n_samples)
    employment_months = base_emp_months.copy()
    employment_months[(employment_type == "unemployed")] = rng.integers(0, 6, size=(employment_type == "unemployed").sum())
    employment_months[(employment_type == "contract")] = rng.integers(0, 60, size=(employment_type == "contract").sum())

    # 10) requested_amount
    requested_amount = rng.normal(loc=30000, scale=12000, size=n_samples)
    requested_amount = np.clip(requested_amount, 5000, 80000)

    # 11) loan_term_months (12,24,36,48,60)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.1, 0.2, 0.35, 0.2, 0.15])

    # 12) interest_rate (%)
    interest_rate = rng.normal(loc=18, scale=5, size=n_samples)
    interest_rate = np.clip(interest_rate, 8, 35)

    # 13) installment – aproximada (no hace falta precisión financiera)
    # cuota aproximada: (requested_amount / loan_term) * factor de interés simplificado
    interest_factor = 1 + (interest_rate / 100 * loan_term_months / 12)  # simple
    installment = (requested_amount * interest_factor) / loan_term_months

    # 14) debt_to_income
    # asumimos deuda total aproximada = cuota * 3 + algo de ruido
    approx_total_debt = installment * 3 + rng.normal(0, 500, size=n_samples)
    approx_total_debt = np.clip(approx_total_debt, 0, None)
    debt_to_income = approx_total_debt / monthly_income
    debt_to_income = np.clip(debt_to_income, 0, 1.5)

    # 15) num_open_loans
    num_open_loans = rng.integers(0, 11, size=n_samples)

    # 16) num_credit_cards
    num_credit_cards = rng.integers(0, 9, size=n_samples)

    # 17) has_mortgage
    has_mortgage = rng.binomial(1, 0.3, size=n_samples)

    # 18) channel
    channel = rng.choice(
        ["branch", "web", "partner", "call_center"],
        size=n_samples,
        p=[0.4, 0.25, 0.25, 0.10],
    )

    # 19) region
    region = rng.choice(
        ["capital", "north", "south", "east", "west"],
        size=n_samples,
        p=[0.4, 0.2, 0.15, 0.15, 0.1],
    )

    # 20) default_90d – generamos con un modelo logístico sintético
    # Construimos un "score" con algunas reglas razonables
    score = (
        -2.0
        + 1.8 * debt_to_income
        + 0.6 * (num_open_loans > 4)
        + 0.4 * (num_credit_cards > 4)
        + 0.8 * (employment_type == "unemployed")
        + 0.4 * (employment_type == "contract")
        + 0.3 * (channel == "web")
        + 0.3 * (channel == "partner")
        + 0.5 * (monthly_income < 2000)
        - 0.3 * (has_mortgage == 1)
        - 0.2 * ((age >= 30) & (age <= 50))
    )

    # Probabilidad usando sigmoide
    p_default = 1 / (1 + np.exp(-score))

    # un poco de ruido aleatorio para que no sea tan determinístico
    p_default = np.clip(p_default + rng.normal(0, 0.02, size=n_samples), 0.01, 0.95)

    default_90d = rng.binomial(1, p_default, size=n_samples)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "application_date": application_date,
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "dependents": dependents,
            "monthly_income": monthly_income.round(2),
            "employment_type": employment_type,
            "employment_months": employment_months,
            "requested_amount": requested_amount.round(2),
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate.round(2),
            "installment": installment.round(2),
            "debt_to_income": debt_to_income.round(3),
            "num_open_loans": num_open_loans,
            "num_credit_cards": num_credit_cards,
            "has_mortgage": has_mortgage,
            "channel": channel,
            "region": region,
            "default_90d": default_90d,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_credit_risk_dataset(n_samples=15000, random_state=42)
    df.to_csv("data/raw/credit_risk_synthetic.csv", index=False)
    print(df.head())
    print(df["default_90d"].mean())
