"""Pydantic data schemas for API requests, responses, and validation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """The 11 customer-level features used by the CLV model."""

    customer_id: Optional[str] = Field(None, description="Optional customer identifier.")
    recency_days: float = Field(..., ge=0, description="Days since customer's most recent purchase.")
    frequency: int = Field(..., ge=1, description="Number of distinct purchase orders/invoices.")
    monetary_total: float = Field(..., ge=0, description="Total historical spending (£).")
    monetary_avg_order: float = Field(..., ge=0, description="Average monetary spend per order (£).")
    tenure_days: float = Field(..., ge=0, description="Days since customer's first observed purchase.")
    avg_days_between_orders: float = Field(..., ge=0, description="Average interval between orders in days.")
    active_months: int = Field(..., ge=1, description="Number of distinct calendar months with activity.")
    average_monthly_spend: float = Field(..., ge=0, description="Historic spend per month of tenure (£).")
    n_unique_products: int = Field(..., ge=1, description="Count of distinct product stock codes bought.")
    avg_basket_size: float = Field(..., ge=0, description="Average quantity of items per order.")
    has_single_purchase: int = Field(
        ..., ge=0, le=1, description="Binary indicator: 1 if customer only bought once, 0 otherwise."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "14646",
                "recency_days": 12.0,
                "frequency": 18,
                "monetary_total": 4520.50,
                "monetary_avg_order": 251.14,
                "tenure_days": 320.0,
                "avg_days_between_orders": 18.82,
                "active_months": 8,
                "average_monthly_spend": 430.52,
                "n_unique_products": 125,
                "avg_basket_size": 42.5,
                "has_single_purchase": 0,
            }
        }
    )


class PredictionResult(BaseModel):
    """Prediction outcome for a single customer."""

    customer_id: Optional[str] = Field(None, description="Customer ID, if provided.")
    predicted_clv_90d: float = Field(..., description="Forecasted gross spend over next 90 days in £.")
    currency: str = Field("GBP", description="Monetary currency code.")
    value_tier: str = Field(..., description="Business value tier (e.g. VIP, High Growth, At-Risk).")
    estimated_decile: int = Field(..., description="Estimated customer decile (1 = top 10%).")
    actionable_recommendation: str = Field(..., description="Strategic marketing/retention recommendation.")


class BatchPredictionRequest(BaseModel):
    """Batch scoring request."""

    customers: List[CustomerFeatures] = Field(..., min_length=1, description="List of customer records.")


class BatchPredictionResponse(BaseModel):
    """Batch scoring summary and per-customer results."""

    total_customers: int
    total_predicted_revenue_90d: float
    average_predicted_clv_90d: float
    high_value_customers_count: int
    predictions: List[PredictionResult]


class RawTransactionItem(BaseModel):
    """A single raw historical transaction line item."""

    Invoice: str = Field(..., description="Invoice number, e.g. '536365'. Returns starting with 'C' are handled.")
    StockCode: str = Field(..., description="Product stock code, e.g. '85123A'.")
    Description: Optional[str] = Field(None, description="Product description.")
    Quantity: int = Field(..., description="Units purchased.")
    Price: float = Field(..., description="Unit price in £.")
    InvoiceDate: str = Field(..., description="Timestamp in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Invoice": "536365",
                "StockCode": "85123A",
                "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "Quantity": 6,
                "Price": 2.55,
                "InvoiceDate": "2011-05-15 10:30:00",
            }
        }
    )


class CustomerTransactionsRequest(BaseModel):
    """Payload to derive features dynamically from transactions and predict."""

    customer_id: Optional[str] = Field(None, description="Customer identifier.")
    as_of_date: Optional[str] = Field(
        None, description="Cutoff date (YYYY-MM-DD). If omitted, defaults to 1 day after the latest transaction."
    )
    transactions: List[RawTransactionItem] = Field(..., min_length=1, description="List of purchase transactions.")


class CustomerTransactionsResponse(BaseModel):
    """Output containing calculated features and prediction."""

    customer_id: Optional[str] = None
    derived_features: Dict[str, float]
    prediction: PredictionResult


class ModelMetadataResponse(BaseModel):
    """Detailed model architecture and parameter metadata."""

    model_name: str
    target_metric: str
    prediction_horizon_days: int
    cv_mae_score: Optional[float]
    feature_count: int
    features: List[str]
    hyperparameters: Dict[str, Any]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    feature_count: int
    version: str
