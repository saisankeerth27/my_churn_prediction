"""
app.py
------
Streamlit app: takes customer details as input via a form and
predicts churn probability using the saved model.

Run locally with:
    streamlit run app/app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this whole project to a public GitHub repo
    2. Go to https://share.streamlit.io -> New app
    3. Point it to app/app.py in your repo
    4. It builds and gives you a live public URL to put on your resume
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import full_preprocess_pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    fit_columns = joblib.load(os.path.join(MODEL_DIR, "fit_columns.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, scaler, fit_columns, model_name


st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to predict their probability of churning. "
    "Built with Logistic Regression / Random Forest / XGBoost (best model auto-selected), "
    "trained on Telco-style customer data."
)

try:
    model, scaler, fit_columns, model_name = load_artifacts()
    st.caption(f"Serving model: **{model_name}**")
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python src/train.py` first to train and save the model.")
    st.stop()

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }])

    X, _, _, _ = full_preprocess_pipeline(input_df, fit_columns=fit_columns, scaler=scaler)
    proba = model.predict_proba(X)[0][1]
    prediction = "Yes" if proba >= 0.5 else "No"

    st.divider()
    if prediction == "Yes":
        st.error(f"⚠️ High churn risk — predicted probability: **{proba:.1%}**")
    else:
        st.success(f"✅ Low churn risk — predicted probability: **{proba:.1%}**")

    st.progress(min(int(proba * 100), 100))
    st.caption(
        "This is a probability, not a certainty — use it to prioritize retention "
        "outreach (e.g. offer a discount or contract upgrade to high-risk customers)."
    )
