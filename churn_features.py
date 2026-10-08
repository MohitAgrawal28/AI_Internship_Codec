"""Shared feature engineering for the churn project.

Kept in its own module so the pickled pipeline can be re-loaded by the
Streamlit app (joblib stores a reference to `churn_features.add_features`).
Assumes the public Telco-style subscription dataset (19 raw input columns).
"""
import numpy as np
import pandas as pd

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

RAW_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", *ADDON_COLS,
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

NUM_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_spend_per_month", "addon_count",
]

CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", *ADDON_COLS, "Contract", "PaperlessBilling",
    "PaymentMethod", "tenure_bucket",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges and add the derived features described in the report:
    tenure bucket, average spend per month, and add-on service count."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 1000],
        labels=["0-12m", "13-24m", "25-48m", "49-60m", "61m+"],
    ).astype(str)

    tenure = df["tenure"].astype(float)
    df["avg_spend_per_month"] = np.where(
        tenure > 0, df["TotalCharges"] / tenure.where(tenure > 0), df["MonthlyCharges"]
    )
    df["addon_count"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    return df
