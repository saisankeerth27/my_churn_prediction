"""
eda.py
------
Exploratory Data Analysis for the churn dataset.
Run this FIRST to understand the data before touching any model.

This prints key stats to the console and saves plots to outputs/eda/
so you can drop them straight into your README or a slide deck.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUT_DIR = "/home/claude/churn-prediction/outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path="/home/claude/churn-prediction/data/telco_churn.csv"):
    df = pd.read_csv(path)
    # TotalCharges sometimes arrives as a string with blank values in the real dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def basic_overview(df):
    print("=" * 60)
    print("SHAPE:", df.shape)
    print("\nNULLS PER COLUMN:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nCHURN DISTRIBUTION:")
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).round(3) * 100, "%")
    print("=" * 60)


def plot_churn_distribution(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", palette=["#4C72B0", "#DD8452"])
    plt.title("Churn Distribution (Target Balance)")
    plt.savefig(f"{OUT_DIR}/churn_distribution.png", bbox_inches="tight", dpi=120)
    plt.close()


def plot_churn_by_contract(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", palette=["#4C72B0", "#DD8452"])
    plt.title("Churn by Contract Type")
    plt.xticks(rotation=15)
    plt.savefig(f"{OUT_DIR}/churn_by_contract.png", bbox_inches="tight", dpi=120)
    plt.close()


def plot_tenure_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True,
                 palette=["#4C72B0", "#DD8452"], element="step")
    plt.title("Tenure Distribution by Churn")
    plt.savefig(f"{OUT_DIR}/tenure_distribution.png", bbox_inches="tight", dpi=120)
    plt.close()


def plot_monthly_charges(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#4C72B0", "#DD8452"])
    plt.title("Monthly Charges by Churn")
    plt.savefig(f"{OUT_DIR}/monthly_charges_by_churn.png", bbox_inches="tight", dpi=120)
    plt.close()


def plot_internet_service(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="InternetService", hue="Churn", palette=["#4C72B0", "#DD8452"])
    plt.title("Churn by Internet Service Type")
    plt.savefig(f"{OUT_DIR}/churn_by_internet_service.png", bbox_inches="tight", dpi=120)
    plt.close()


def print_key_insights(df):
    print("\nKEY INSIGHTS (put these in your README):")
    mtm_churn = df[df["Contract"] == "Month-to-month"]["Churn"].eq("Yes").mean()
    two_yr_churn = df[df["Contract"] == "Two year"]["Churn"].eq("Yes").mean()
    print(f"- Month-to-month churn rate: {mtm_churn:.1%} vs Two-year: {two_yr_churn:.1%}")

    fiber_churn = df[df["InternetService"] == "Fiber optic"]["Churn"].eq("Yes").mean()
    dsl_churn = df[df["InternetService"] == "DSL"]["Churn"].eq("Yes").mean()
    print(f"- Fiber optic churn rate: {fiber_churn:.1%} vs DSL: {dsl_churn:.1%}")

    low_tenure_churn = df[df["tenure"] < 12]["Churn"].eq("Yes").mean()
    high_tenure_churn = df[df["tenure"] > 48]["Churn"].eq("Yes").mean()
    print(f"- Customers with <12 months tenure churn: {low_tenure_churn:.1%} vs >48 months: {high_tenure_churn:.1%}")


if __name__ == "__main__":
    df = load_data()
    basic_overview(df)
    plot_churn_distribution(df)
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    plot_monthly_charges(df)
    plot_internet_service(df)
    print_key_insights(df)
    print(f"\nPlots saved to {OUT_DIR}/")
