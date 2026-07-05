"""
train.py
--------
Trains and compares 3 models: Logistic Regression, Random Forest, XGBoost.
Handles class imbalance with SMOTE. Evaluates with the metrics that
actually matter for churn (recall, F1, ROC-AUC -- not just accuracy).
Saves the best model + its supporting artifacts for the Streamlit app.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import full_preprocess_pipeline

DATA_PATH = "/home/claude/churn-prediction/data/telco_churn.csv"
MODEL_DIR = "/home/claude/churn-prediction/models"
OUT_DIR = "/home/claude/churn-prediction/outputs/train"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X, y, fit_columns, scaler = full_preprocess_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, fit_columns, scaler


def apply_smote(X_train, y_train):
    print(f"Before SMOTE -> Class counts: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE  -> Class counts: {y_res.value_counts().to_dict()}")
    return X_res, y_res


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC_AUC": roc_auc_score(y_test, proba),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.3f}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    return metrics, preds


def plot_confusion_matrix(name, y_test, preds):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{OUT_DIR}/confusion_matrix_{name.replace(' ', '_')}.png",
                bbox_inches="tight", dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4C72B0")
    plt.title(f"Top 15 Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance_{model_name.replace(' ', '_')}.png", dpi=120)
    plt.close()
    print(f"\nTop 5 features for {model_name}:")
    print(imp_df.head(5).to_string(index=False))


def main():
    X_train, X_test, y_train, y_test, fit_columns, scaler = load_and_split()
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="logloss", random_state=42
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        metrics, preds = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model
        plot_confusion_matrix(name, y_test, preds)

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by ROC-AUC):")
    print(results_df.to_string(index=False))
    print("=" * 60)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    plot_feature_importance(best_model, fit_columns, best_model_name)

    # Save everything the app needs to reproduce this exact model at inference time
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(fit_columns, f"{MODEL_DIR}/fit_columns.pkl")
    joblib.dump(best_model_name, f"{MODEL_DIR}/best_model_name.pkl")
    results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

    print(f"\nSaved best model ({best_model_name}) and artifacts to {MODEL_DIR}/")
    print(f"Saved evaluation outputs to {OUT_DIR}/")


if __name__ == "__main__":
    main()
