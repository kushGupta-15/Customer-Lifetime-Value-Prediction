"""Cutoff-parameterised target construction and customer-level feature engineering.

The cutoff is the single lever that separates past from future. Every feature is
computed from transactions strictly before it, and the target exclusively from
transactions after it, so the two can never overlap by construction rather than
by convention.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import DATA_DIR

CUTOFF_DATE = pd.Timestamp("2011-09-01")
PREDICTION_WINDOW_DAYS = 90

CUSTOMER_DATASET_PATH = DATA_DIR / "customer_features.csv"

# Average days per month (365.25 / 12), used to express tenure in months.
DAYS_PER_MONTH = 30.4375

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg_order",
    "tenure_days",
    "avg_days_between_orders",
    "active_months",
    "average_monthly_spend",
    "n_unique_products",
    "avg_basket_size",
    "has_single_purchase",
]
TARGET_COLUMN = "future_clv"


def split_windows(df, cutoff=CUTOFF_DATE, horizon_days=PREDICTION_WINDOW_DAYS):
    """Split transactions into the observation and prediction windows.

    Strict inequality on the cutoff is what makes the split leakage-free: a
    transaction on the cutoff timestamp itself belongs to neither window's
    ambiguous middle - it falls into the future.
    """
    observation = df[df["InvoiceDate"] < cutoff]
    prediction_end = cutoff + pd.Timedelta(days=horizon_days)
    prediction = df[(df["InvoiceDate"] >= cutoff) & (df["InvoiceDate"] < prediction_end)]
    return observation, prediction


def build_target(prediction_df, customer_index):
    """Sum future spend per customer over the prediction window.

    Reindexing onto the observation-window population and filling with 0 is the
    point of the exercise: customers who buy nothing in the next 90 days are the
    majority class and carry real business meaning, so they are kept, not dropped.
    """
    future_spend = prediction_df.groupby("Customer ID")["line_revenue"].sum()
    return future_spend.reindex(customer_index).fillna(0.0).rename(TARGET_COLUMN)


def build_features(observation_df, cutoff=CUTOFF_DATE):
    """Build the 11 customer-level features from observation-window data only."""
    grouped = observation_df.groupby("Customer ID")

    # RFM core. Frequency counts distinct invoices rather than line items, so a
    # single large basket does not read as many purchases.
    agg = grouped.agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        frequency=("Invoice", "nunique"),
        monetary_total=("line_revenue", "sum"),
        total_quantity=("Quantity", "sum"),
        n_unique_products=("StockCode", "nunique"),
    )

    features = pd.DataFrame(index=agg.index)
    features["recency_days"] = (cutoff - agg["last_purchase"]).dt.days
    features["frequency"] = agg["frequency"]
    features["monetary_total"] = agg["monetary_total"]
    features["monetary_avg_order"] = agg["monetary_total"] / agg["frequency"]

    # Tenure and temporal cadence.
    features["tenure_days"] = (cutoff - agg["first_purchase"]).dt.days
    # For one-off buyers there is no gap to measure, so tenure stands in as the
    # longest interval consistent with what we have observed.
    features["avg_days_between_orders"] = np.where(
        agg["frequency"] > 1,
        features["tenure_days"] / (agg["frequency"] - 1),
        features["tenure_days"],
    )
    # Distinct calendar months with activity - a spread-out buyer and a one-week
    # burst can share a frequency but differ sharply here.
    features["active_months"] = grouped["InvoiceDate"].apply(
        lambda s: s.dt.to_period("M").nunique()
    )
    tenure_months = (features["tenure_days"] / DAYS_PER_MONTH).clip(lower=1.0)
    features["average_monthly_spend"] = agg["monetary_total"] / tenure_months

    # Behavioural breadth and basket size.
    features["n_unique_products"] = agg["n_unique_products"]
    features["avg_basket_size"] = agg["total_quantity"] / agg["frequency"]
    # One-off buyers behave differently enough to be worth an explicit flag that
    # tree models can split on directly.
    features["has_single_purchase"] = (agg["frequency"] == 1).astype("int64")

    return features[FEATURE_COLUMNS]


def build_customer_dataset(df, cutoff=CUTOFF_DATE, horizon_days=PREDICTION_WINDOW_DAYS):
    """Assemble the modelling table: one row per customer, 11 features + target.

    Population is defined by the observation window alone, so the feature table
    never gains or loses a customer on the basis of future behaviour.
    """
    observation, prediction = split_windows(df, cutoff, horizon_days)
    features = build_features(observation, cutoff)
    target = build_target(prediction, features.index)
    return features.join(target)


def add_log_target(dataset, target_column=TARGET_COLUMN):
    """Attach the log1p-transformed target used for training.

    future_clv spans zero to six figures and is dominated by zeros; log1p pulls
    that into a range where squared-error training is not driven entirely by a
    handful of very large customers. log1p (not log) because zero is legitimate.
    """
    return dataset.assign(log_future_clv=np.log1p(dataset[target_column]))


def load_customer_dataset(path=CUSTOMER_DATASET_PATH):
    """Read the customer-level table written by notebook 02."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Customer dataset not found at {path}. "
            "Run notebooks/02_features_and_target.ipynb first."
        )
    return pd.read_csv(path, index_col="Customer ID")
