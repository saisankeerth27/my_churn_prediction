"""
preprocess.py
-------------
Cleaning + feature engineering, kept in its own module so both
train.py and app.py can reuse the exact same transformation logic
(this consistency matters a lot in real jobs -- train/serve skew is
a classic bug when preprocessing lives in two different places).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple derived features that tend to help tree models and also
    give you good talking points in interviews."""
    df = df.copy()

    # Bucket tenure into readable groups
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"], include_lowest=True
    )

    # Count how many "add-on" services a customer has subscribed to
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addon_services"] = sum((df[c] == "Yes").astype(int) for c in addon_cols)

    # Average monthly spend so far (avoids leakage since it's derived from tenure/total)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df


# Fixed category lists so a single-row prediction (e.g. one customer in the
# Streamlit app) produces the SAME dummy columns as training on thousands of
# rows. Without this, pd.get_dummies on 1 row only creates a column for
# whatever category is present, silently zeroing out real signal -- a classic
# train/serve skew bug.
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                      "Electronic check", "Mailed check"],
    "tenure_group": ["0-12", "13-24", "25-48", "49-72"],
}


def encode_features(df: pd.DataFrame, fit_columns=None):
    """One-hot encode categoricals using FIXED category lists (see
    CATEGORY_VALUES above), so encoding is identical whether we're
    processing 7,000 training rows or 1 row from the app form."""
    df = df.copy()
    cat_cols = CATEGORICAL_COLS + ["tenure_group"]

    for col in cat_cols:
        df[col] = pd.Categorical(df[col].astype(str), categories=CATEGORY_VALUES[col])

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[fit_columns]

    return df_encoded


def full_preprocess_pipeline(df: pd.DataFrame, fit_columns=None, scaler: StandardScaler = None):
    """
    Runs the full pipeline: clean -> feature engineer -> encode -> scale.
    Returns (X_processed, fit_columns_used, fitted_scaler)
    so train.py can save these and app.py can reuse them exactly.
    """
    df = clean_data(df)
    y = None
    if "Churn" in df.columns:
        y = (df["Churn"] == "Yes").astype(int)
        df = df.drop(columns=["Churn"])

    df = add_features(df)
    X = encode_features(df, fit_columns=fit_columns)

    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend"]
    scale_cols = [c for c in scale_cols if c in X.columns]

    if scaler is None:
        scaler = StandardScaler()
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        X[scale_cols] = scaler.transform(X[scale_cols])

    return X, y, list(X.columns), scaler
