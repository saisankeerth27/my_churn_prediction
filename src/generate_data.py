"""
generate_data.py
----------------
Generates a synthetic customer churn dataset that mirrors the structure and
statistical patterns of the well-known Telco Customer Churn dataset
(IBM/Kaggle). Use this to run the whole pipeline end-to-end for practice.

NOTE: If you want to use the REAL Telco Customer Churn dataset instead
(recommended for your final resume project, since recruiters may recognize
it), download it from Kaggle ("Telco Customer Churn" by blastchar) and
place it at data/telco_churn.csv with the same column names used below.
This script exists so you can test-run the entire pipeline right now
without needing to download anything first.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7043  # same size as the real Telco dataset

def generate_customer_data(n=N):
    df = pd.DataFrame()
    df["customerID"] = [f"{i:04d}-CUST" for i in range(n)]

    df["gender"] = np.random.choice(["Male", "Female"], n)
    df["SeniorCitizen"] = np.random.choice([0, 1], n, p=[0.84, 0.16])
    df["Partner"] = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    df["Dependents"] = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    df["tenure"] = np.random.randint(0, 73, n)  # months, 0-72

    df["PhoneService"] = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service",
        np.random.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )

    df["InternetService"] = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    def dependent_internet_col(p_yes):
        return np.where(
            df["InternetService"] == "No", "No internet service",
            np.random.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        )

    df["OnlineSecurity"] = dependent_internet_col(0.29)
    df["OnlineBackup"] = dependent_internet_col(0.34)
    df["DeviceProtection"] = dependent_internet_col(0.34)
    df["TechSupport"] = dependent_internet_col(0.29)
    df["StreamingTV"] = dependent_internet_col(0.38)
    df["StreamingMovies"] = dependent_internet_col(0.39)

    df["Contract"] = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
    )
    df["PaperlessBilling"] = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    df["PaymentMethod"] = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges depend loosely on services subscribed
    base = 18 + df["tenure"] * 0.0
    internet_add = df["InternetService"].map({"DSL": 15, "Fiber optic": 45, "No": 0})
    extra_services = sum(
        (df[col] == "Yes").astype(int) * 5
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    )
    noise = np.random.normal(0, 5, n)
    df["MonthlyCharges"] = np.clip(base + internet_add + extra_services + noise, 18, 120).round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"] + np.random.normal(0, 20, n)).clip(lower=0).round(2)

    # ---- Churn logic: built to reflect realistic real-world patterns ----
    # Month-to-month, fiber optic, high monthly charges, low tenure, no tech support
    # all increase churn probability (this mirrors well-documented real findings)
    churn_score = (
        (df["Contract"] == "Month-to-month") * 0.35
        + (df["Contract"] == "One year") * 0.10
        + (df["InternetService"] == "Fiber optic") * 0.20
        + (df["TechSupport"] == "No") * 0.10
        + (df["tenure"] < 12) * 0.25
        + (df["MonthlyCharges"] > 80) * 0.15
        + (df["PaperlessBilling"] == "Yes") * 0.05
        + (df["PaymentMethod"] == "Electronic check") * 0.10
        - (df["Contract"] == "Two year") * 0.30
        - (df["tenure"] > 48) * 0.20
        + np.random.normal(0, 0.15, n)
    )
    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.55)))  # sigmoid squashing, tuned for ~27% churn
    df["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    df.to_csv("/home/claude/churn-prediction/data/telco_churn.csv", index=False)
    print(f"Generated {len(df)} rows.")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(df.head())
