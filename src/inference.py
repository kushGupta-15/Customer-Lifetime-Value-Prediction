"""Inference and serving engine for the CLV model.

Handles model loading, feature extraction from raw transactions,
monetary predictions, and business segment assignment.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data_prep import NON_PRODUCT_CODES, normalize_stock_code
from src.features import DAYS_PER_MONTH, FEATURE_COLUMNS
from src.modeling import BEST_PARAMS_PATH, FINAL_MODEL_PATH

logger = logging.getLogger("clv.inference")


class CLVPredictor:
    """Predictor class managing the trained CLV model and scoring logic."""

    def __init__(self, model_path: Optional[Path] = None, params_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else FINAL_MODEL_PATH
        self.params_path = Path(params_path) if params_path else BEST_PARAMS_PATH
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load model artifact and metadata, bootstrapping if necessary."""
        # Load metadata
        if self.params_path.exists():
            try:
                with open(self.params_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", self.params_path, e)

        # Load model artifact
        if self.model_path.exists():
            logger.info("Loading CLV model from %s", self.model_path)
            self.model = joblib.load(self.model_path)
        else:
            logger.warning("Model file %s not found. Bootstrapping initial model...", self.model_path)
            from src.train import train

            self.model, _ = train(force_synthetic=True, output_path=self.model_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @staticmethod
    def assign_tier(predicted_clv: float) -> Tuple[str, int]:
        """Categorize customer into business tiers and estimated deciles.

        Derived from decile lift analysis in the test set:
        - Decile 1 (Top ~10%): Mean £1,640+ (Captures 54% of revenue) -> VIP Tier
        - Decile 2-3 (Next ~20%): £60 - £500+ -> High Growth Tier
        - Decile 4-6: £5 - £60 -> Moderate Value Tier
        - Decile 7-10: £0 - £5 -> Low / At-Risk Tier
        """
        val = max(0.0, float(predicted_clv))
        if val >= 1000.0:
            return "Tier 1: VIP / High Value", 1
        elif val >= 150.0:
            return "Tier 2: Core / High Growth", 2
        elif val >= 30.0:
            return "Tier 3: Moderate Value", 4
        elif val >= 5.0:
            return "Tier 4: Low Value", 6
        else:
            return "Tier 5: At-Risk / Dormant", 8

    def predict(self, features_df: pd.DataFrame) -> List[float]:
        """Generate monetary predictions (£) for a DataFrame containing the 11 features."""
        if not self.is_ready:
            raise RuntimeError("Model is not initialized.")

        # Ensure all required features are present
        missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        X = features_df[FEATURE_COLUMNS]
        # Invert log1p transform and clip at zero
        raw_preds = self.model.predict(X)
        monetary_preds = np.clip(np.expm1(raw_preds), 0.0, None)
        return [round(float(p), 2) for p in monetary_preds]

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict CLV for a single customer feature dictionary."""
        df = pd.DataFrame([features])
        pred_val = self.predict(df)[0]
        tier, decile = self.assign_tier(pred_val)
        return {
            "predicted_clv_90d": pred_val,
            "value_tier": tier,
            "estimated_decile": decile,
            "features_used": {col: float(features[col]) for col in FEATURE_COLUMNS if col in features},
        }

    def predict_batch(self, customer_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict CLV for a batch of customers."""
        df = pd.DataFrame(customer_records)
        preds = self.predict(df)
        results = []
        for i, pred_val in enumerate(preds):
            customer_id = customer_records[i].get("customer_id")
            tier, decile = self.assign_tier(pred_val)
            results.append(
                {
                    "customer_id": customer_id,
                    "predicted_clv_90d": pred_val,
                    "value_tier": tier,
                    "estimated_decile": decile,
                }
            )
        return results

    @staticmethod
    def extract_features_from_transactions(
        transactions: List[Dict[str, Any]],
        as_of_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, float]:
        """Compute the 11 features from a list of raw transaction dictionaries.

        Applies data cleaning logic:
        - Drops cancellations (Invoice starting with 'C' or Quantity <= 0)
        - Drops Price <= 0
        - Normalizes and removes non-product codes
        - Computes line_revenue = Quantity * Price
        """
        if not transactions:
            raise ValueError("No transactions provided.")

        raw_df = pd.DataFrame(transactions)
        # Required raw fields
        for field in ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]:
            if field not in raw_df.columns:
                raise ValueError(f"Transaction missing required field: {field}")

        df = raw_df.copy()
        df["Invoice"] = df["Invoice"].astype(str)
        df["StockCode"] = normalize_stock_code(df["StockCode"])
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Cleaning filters matching src.data_prep
        df = df[~df["Invoice"].str.startswith("C")]
        df = df[df["Quantity"] > 0]
        df = df[df["Price"] > 0]
        df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

        if df.empty:
            raise ValueError("No valid purchases remaining after filtering returns/adjustments.")

        df["line_revenue"] = df["Quantity"] * df["Price"]

        # Determine cutoff date
        if as_of_date is not None:
            cutoff = pd.to_datetime(as_of_date)
            # Filter transactions before cutoff
            df = df[df["InvoiceDate"] < cutoff]
            if df.empty:
                raise ValueError("No transactions found strictly before the specified as_of_date cutoff.")
        else:
            # Cutoff defaults to 1 day after the latest purchase
            cutoff = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        last_purchase = df["InvoiceDate"].max()
        first_purchase = df["InvoiceDate"].min()
        frequency = int(df["Invoice"].nunique())
        monetary_total = float(df["line_revenue"].sum())
        total_quantity = float(df["Quantity"].sum())
        n_unique_products = int(df["StockCode"].nunique())

        recency_days = float((cutoff - last_purchase).days)
        tenure_days = float((cutoff - first_purchase).days)
        monetary_avg_order = monetary_total / frequency

        avg_days_between_orders = (
            tenure_days / (frequency - 1) if frequency > 1 else tenure_days
        )
        active_months = int(df["InvoiceDate"].dt.to_period("M").nunique())
        tenure_months = max(tenure_days / DAYS_PER_MONTH, 1.0)
        average_monthly_spend = monetary_total / tenure_months

        avg_basket_size = total_quantity / frequency
        has_single_purchase = 1 if frequency == 1 else 0

        features = {
            "recency_days": round(recency_days, 2),
            "frequency": frequency,
            "monetary_total": round(monetary_total, 2),
            "monetary_avg_order": round(monetary_avg_order, 2),
            "tenure_days": round(tenure_days, 2),
            "avg_days_between_orders": round(avg_days_between_orders, 2),
            "active_months": active_months,
            "average_monthly_spend": round(average_monthly_spend, 2),
            "n_unique_products": n_unique_products,
            "avg_basket_size": round(avg_basket_size, 2),
            "has_single_purchase": has_single_purchase,
        }
        return features


# Default singleton instance
_default_predictor: Optional[CLVPredictor] = None


def get_predictor() -> CLVPredictor:
    """Return singleton predictor instance."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = CLVPredictor()
    return _default_predictor
