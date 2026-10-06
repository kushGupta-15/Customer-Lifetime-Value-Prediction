"""Standalone training script for the CLV Random Forest model.

Trains the model using best hyperparameters from models/best_params.json and
serializes it to models/final_model.joblib.

Can run against:
1. Pre-computed customer features (data/customer_features.csv)
2. Cleaned transactions (data/transactions_clean.csv)
3. Raw Excel workbook (data/online_retail_II.xlsx)
4. Synthetic benchmark data (via --synthetic flag or automatic fallback for testing)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import (
    CLEAN_TRANSACTIONS_PATH,
    DATA_DIR,
    RAW_EXCEL_PATH,
    clean_transactions,
    load_clean_transactions,
    load_raw_transactions,
)
from src.features import (
    CUSTOMER_DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_customer_dataset,
    load_customer_dataset,
)
from src.modeling import (
    BEST_PARAMS_PATH,
    FINAL_MODEL_PATH,
    RANDOM_SEED,
    fit_on_log_target,
    predict_monetary,
    regression_metrics,
    save_model,
    split_train_test,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clv.train")


def load_hyperparameters(path=BEST_PARAMS_PATH):
    """Load tuned parameters from best_params.json or return defaults."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info("Loaded hyperparameters from %s (CV MAE: %.2f)", path, data.get("cv_mae", 0.0))
            return data["params"]
    logger.warning("Parameter file %s not found. Using default parameters.", path)
    return {
        "n_estimators": 450,
        "max_depth": 22,
        "min_samples_leaf": 3,
        "min_samples_split": 3,
        "max_features": 0.782,
    }


def generate_synthetic_dataset(n_customers=500, random_state=RANDOM_SEED):
    """Generate realistic synthetic customer feature data for dry-run/testing."""
    rng = np.random.default_rng(random_state)
    logger.info("Generating %d synthetic customer records for testing...", n_customers)

    recency_days = rng.integers(1, 365, size=n_customers)
    frequency = rng.geometric(p=0.25, size=n_customers)
    monetary_avg_order = rng.lognormal(mean=4.0, sigma=0.8, size=n_customers)
    monetary_total = frequency * monetary_avg_order
    tenure_days = recency_days + rng.integers(0, 365, size=n_customers)

    avg_days_between_orders = np.where(
        frequency > 1, tenure_days / np.maximum(frequency - 1, 1), tenure_days
    )
    active_months = np.clip(np.ceil(frequency / 2), 1, 24).astype(int)
    tenure_months = np.maximum(tenure_days / 30.4375, 1.0)
    average_monthly_spend = monetary_total / tenure_months

    n_unique_products = np.clip(frequency * rng.integers(1, 10, size=n_customers), 1, 500)
    avg_basket_size = rng.lognormal(mean=2.5, sigma=0.6, size=n_customers)
    has_single_purchase = (frequency == 1).astype(int)

    # Realistic future spend with zero-inflation (~55% zero buyers)
    will_buy = (recency_days < 120) & (rng.random(size=n_customers) > 0.35)
    future_spend = np.where(
        will_buy,
        np.exp(rng.normal(np.log(average_monthly_spend * 3 + 1), 0.5)),
        0.0,
    )

    df = pd.DataFrame(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg_order": monetary_avg_order,
            "tenure_days": tenure_days,
            "avg_days_between_orders": avg_days_between_orders,
            "active_months": active_months,
            "average_monthly_spend": average_monthly_spend,
            "n_unique_products": n_unique_products,
            "avg_basket_size": avg_basket_size,
            "has_single_purchase": has_single_purchase,
            TARGET_COLUMN: future_spend,
        },
        index=pd.Index(range(10001, 10001 + n_customers), name="Customer ID"),
    )
    return df


def get_training_dataset(force_synthetic=False):
    """Retrieve or build the customer dataset from available sources."""
    if force_synthetic:
        return generate_synthetic_dataset()

    # 1. Customer features table
    if CUSTOMER_DATASET_PATH.exists():
        logger.info("Loading existing customer dataset from %s", CUSTOMER_DATASET_PATH)
        return load_customer_dataset()

    # 2. Cleaned transactions CSV
    if CLEAN_TRANSACTIONS_PATH.exists():
        logger.info("Building customer features from %s", CLEAN_TRANSACTIONS_PATH)
        df = load_clean_transactions()
        customer_df = build_customer_dataset(df)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        customer_df.to_csv(CUSTOMER_DATASET_PATH)
        return customer_df

    # 3. Raw Excel file
    if RAW_EXCEL_PATH.exists():
        logger.info("Cleaning raw Excel transactions from %s", RAW_EXCEL_PATH)
        raw_df = load_raw_transactions()
        clean_df, _ = clean_transactions(raw_df)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(CLEAN_TRANSACTIONS_PATH, index=False)
        customer_df = build_customer_dataset(clean_df)
        customer_df.to_csv(CUSTOMER_DATASET_PATH)
        return customer_df

    logger.warning("No data found in %s. Generating synthetic dataset to bootstrap model.", DATA_DIR)
    return generate_synthetic_dataset()


def train(force_synthetic=False, output_path=FINAL_MODEL_PATH):
    """Execute training pipeline and save model artifact."""
    params = load_hyperparameters()
    dataset = get_training_dataset(force_synthetic=force_synthetic)
    logger.info("Dataset loaded with %d customers and %d features.", len(dataset), len(FEATURE_COLUMNS))

    # Split train+val vs test
    X_trainval, X_test, y_trainval, y_test = split_train_test(dataset, random_state=RANDOM_SEED)
    logger.info("Training set size: %d customers, Test set size: %d customers", len(X_trainval), len(X_test))

    model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **params)
    logger.info("Fitting RandomForestRegressor with log1p target...")
    fit_on_log_target(model, X_trainval, y_trainval)

    # Evaluate on test set
    preds = predict_monetary(model, X_test)
    metrics = regression_metrics(y_test, preds)
    logger.info("Hold-out Test Metrics: MAE = £%.2f | RMSE = £%.2f | R2 = %.3f", metrics["MAE"], metrics["RMSE"], metrics["R2"])

    saved_path = save_model(model, path=output_path)
    logger.info("Model successfully saved to %s", saved_path)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and serialize CLV prediction model.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for testing.")
    parser.add_argument("--output", type=str, default=str(FINAL_MODEL_PATH), help="Target joblib output path.")
    args = parser.parse_args()

    train(force_synthetic=args.synthetic, output_path=Path(args.output))
