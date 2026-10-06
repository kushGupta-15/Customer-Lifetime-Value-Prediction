"""FastAPI REST Service for Customer Lifetime Value (CLV) Prediction.

Provides endpoints for online single-customer prediction, high-throughput
batch scoring, on-the-fly transaction feature extraction, and model telemetry.
"""

import logging
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    CustomerTransactionsRequest,
    CustomerTransactionsResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResult,
)
from src.features import FEATURE_COLUMNS, PREDICTION_WINDOW_DAYS
from src.inference import CLVPredictor, get_predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clv.api")

API_VERSION = "1.0.0"


def _generate_recommendation(tier: str, pred_clv: float, recency: float) -> str:
    """Provide domain-driven marketing recommendations based on tier and recency."""
    if "VIP" in tier:
        return "Priority VIP: Assign dedicated retention budget, VIP perks, and high-touch support."
    elif "Core" in tier:
        return "High-Growth: Target with personalized cross-sell recommendations and loyalty incentives."
    elif "Moderate" in tier:
        return "Standard Engagement: Include in regular automated lifecycle and promotional newsletters."
    elif "Low Value" in tier:
        return "Light-Touch: Use cost-effective digital channels; avoid expensive direct marketing."
    else:
        if recency > 180:
            return "Win-Back / Re-engagement: Test targeted win-back discount; suppress if unresponsive."
        return "At-Risk: Monitor next purchase cycle or reallocate acquisition spend."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to pre-warm the model at startup."""
    logger.info("Initializing CLV Predictor service...")
    try:
        predictor = get_predictor()
        if predictor.is_ready:
            logger.info("CLV Model loaded successfully.")
    except Exception as e:
        logger.error("Failed to initialize predictor: %s", e)
    yield
    logger.info("Shutting down CLV Predictor service.")


app = FastAPI(
    title="Customer Lifetime Value (CLV) Prediction API",
    description=(
        "Production-grade microservice predicting customer spend over a **90-day horizon** "
        "from transaction history and RFM/cadence/breadth features."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Enable CORS for cross-origin web/dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Service landing page and navigation guide."""
    return {
        "service": "Customer Lifetime Value (CLV) Prediction API",
        "version": API_VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "model_metadata": "/metadata",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness and readiness health probe."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy" if predictor.is_ready else "degraded",
        model_loaded=predictor.is_ready,
        feature_count=len(FEATURE_COLUMNS),
        version=API_VERSION,
    )


@app.get("/metadata", response_model=ModelMetadataResponse, tags=["Monitoring"])
def model_metadata():
    """Inspect model configuration, training score, and feature list."""
    predictor = get_predictor()
    meta = predictor.metadata
    return ModelMetadataResponse(
        model_name=meta.get("model", "RandomForestRegressor"),
        target_metric="log1p(future_clv) -> expm1 monetary scale (£)",
        prediction_horizon_days=PREDICTION_WINDOW_DAYS,
        cv_mae_score=round(meta.get("cv_mae", 0.0), 2) if "cv_mae" in meta else None,
        feature_count=len(FEATURE_COLUMNS),
        features=FEATURE_COLUMNS,
        hyperparameters=meta.get("params", {}),
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_single_customer(payload: CustomerFeatures):
    """Predict 90-day CLV for a single customer given their 11 features."""
    predictor = get_predictor()
    try:
        data_dict = payload.model_dump()
        result = predictor.predict_single(data_dict)
        pred_val = result["predicted_clv_90d"]
        tier = result["value_tier"]
        decile = result["estimated_decile"]
        rec = _generate_recommendation(tier, pred_val, payload.recency_days)

        return PredictionResult(
            customer_id=payload.customer_id,
            predicted_clv_90d=pred_val,
            currency="GBP",
            value_tier=tier,
            estimated_decile=decile,
            actionable_recommendation=rec,
        )
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_customers(payload: BatchPredictionRequest):
    """Score a batch of customer feature records with summary statistics."""
    predictor = get_predictor()
    try:
        raw_list = [cust.model_dump() for cust in payload.customers]
        df = pd.DataFrame(raw_list)
        preds = predictor.predict(df)

        prediction_results: List[PredictionResult] = []
        high_value_count = 0
        total_revenue = 0.0

        for i, val in enumerate(preds):
            cust = payload.customers[i]
            tier, decile = CLVPredictor.assign_tier(val)
            if "VIP" in tier or "Core" in tier:
                high_value_count += 1
            total_revenue += val
            rec = _generate_recommendation(tier, val, cust.recency_days)

            prediction_results.append(
                PredictionResult(
                    customer_id=cust.customer_id,
                    predicted_clv_90d=val,
                    currency="GBP",
                    value_tier=tier,
                    estimated_decile=decile,
                    actionable_recommendation=rec,
                )
            )

        n = len(prediction_results)
        return BatchPredictionResponse(
            total_customers=n,
            total_predicted_revenue_90d=round(total_revenue, 2),
            average_predicted_clv_90d=round(total_revenue / n, 2) if n > 0 else 0.0,
            high_value_customers_count=high_value_count,
            predictions=prediction_results,
        )
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )


@app.post("/predict/transactions", response_model=CustomerTransactionsResponse, tags=["Inference"])
def predict_from_transactions(payload: CustomerTransactionsRequest):
    """Derive the 11 features from raw customer transactions and predict CLV."""
    predictor = get_predictor()
    try:
        raw_items = [item.model_dump() for item in payload.transactions]
        features = predictor.extract_features_from_transactions(
            transactions=raw_items, as_of_date=payload.as_of_date
        )

        # Score derived features
        pred_dict = predictor.predict_single(features)
        pred_val = pred_dict["predicted_clv_90d"]
        tier = pred_dict["value_tier"]
        decile = pred_dict["estimated_decile"]
        rec = _generate_recommendation(tier, pred_val, features["recency_days"])

        pred_result = PredictionResult(
            customer_id=payload.customer_id,
            predicted_clv_90d=pred_val,
            currency="GBP",
            value_tier=tier,
            estimated_decile=decile,
            actionable_recommendation=rec,
        )

        return CustomerTransactionsResponse(
            customer_id=payload.customer_id,
            derived_features=features,
            prediction=pred_result,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Transaction-based prediction failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transaction inference error: {str(e)}",
        )
