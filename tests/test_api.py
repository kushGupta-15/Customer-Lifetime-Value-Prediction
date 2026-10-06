"""Unit and integration tests for the FastAPI CLV Prediction API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify landing page endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "service" in data


def test_health_check():
    """Verify health probe reports healthy and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True
    assert data["feature_count"] == 11


def test_metadata_endpoint():
    """Verify model metadata endpoint."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "RandomForest" or "RandomForest" in data["model_name"]
    assert len(data["features"]) == 11
    assert data["prediction_horizon_days"] == 90


def test_predict_single_customer():
    """Test single customer prediction with valid feature vector."""
    payload = {
        "customer_id": "TEST_CUST_001",
        "recency_days": 14.0,
        "frequency": 12,
        "monetary_total": 3500.0,
        "monetary_avg_order": 291.67,
        "tenure_days": 300.0,
        "avg_days_between_orders": 27.27,
        "active_months": 7,
        "average_monthly_spend": 355.0,
        "n_unique_products": 85,
        "avg_basket_size": 35.0,
        "has_single_purchase": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST_CUST_001"
    assert data["predicted_clv_90d"] >= 0.0
    assert data["currency"] == "GBP"
    assert "value_tier" in data
    assert 1 <= data["estimated_decile"] <= 10
    assert len(data["actionable_recommendation"]) > 0


def test_predict_validation_error():
    """Test validation reject on illegal feature values (e.g. negative recency)."""
    payload = {
        "recency_days": -5.0,  # Invalid: must be >= 0
        "frequency": 0,        # Invalid: must be >= 1
        "monetary_total": 100.0,
        "monetary_avg_order": 100.0,
        "tenure_days": 10.0,
        "avg_days_between_orders": 10.0,
        "active_months": 1,
        "average_monthly_spend": 10.0,
        "n_unique_products": 5,
        "avg_basket_size": 2.0,
        "has_single_purchase": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch():
    """Test batch scoring of multiple customers."""
    payload = {
        "customers": [
            {
                "customer_id": "VIP_01",
                "recency_days": 5.0,
                "frequency": 25,
                "monetary_total": 8500.0,
                "monetary_avg_order": 340.0,
                "tenure_days": 350.0,
                "avg_days_between_orders": 14.58,
                "active_months": 10,
                "average_monthly_spend": 739.0,
                "n_unique_products": 150,
                "avg_basket_size": 60.0,
                "has_single_purchase": 0,
            },
            {
                "customer_id": "LOW_02",
                "recency_days": 250.0,
                "frequency": 1,
                "monetary_total": 45.0,
                "monetary_avg_order": 45.0,
                "tenure_days": 250.0,
                "avg_days_between_orders": 250.0,
                "active_months": 1,
                "average_monthly_spend": 5.48,
                "n_unique_products": 3,
                "avg_basket_size": 4.0,
                "has_single_purchase": 1,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert data["total_predicted_revenue_90d"] >= 0.0
    # Customer 1 should have higher CLV than customer 2
    pred_1 = data["predictions"][0]["predicted_clv_90d"]
    pred_2 = data["predictions"][1]["predicted_clv_90d"]
    assert pred_1 >= pred_2


def test_predict_from_transactions():
    """Test feature extraction and prediction from raw transaction items."""
    payload = {
        "customer_id": "TX_CUST_99",
        "transactions": [
            {
                "Invoice": "500001",
                "StockCode": "22423",
                "Description": "REGENCY CAKESTAND 3 TIER",
                "Quantity": 3,
                "Price": 12.75,
                "InvoiceDate": "2011-06-01 11:20:00",
            },
            {
                "Invoice": "500002",
                "StockCode": "85123A",
                "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "Quantity": 10,
                "Price": 2.55,
                "InvoiceDate": "2011-07-15 14:10:00",
            },
            {
                "Invoice": "500003",
                "StockCode": "47566",
                "Description": "PARTY BUNTING",
                "Quantity": 5,
                "Price": 4.95,
                "InvoiceDate": "2011-08-20 09:45:00",
            },
            # Return transaction that should be cleaned out
            {
                "Invoice": "C500004",
                "StockCode": "22423",
                "Description": "Discount",
                "Quantity": -1,
                "Price": 12.75,
                "InvoiceDate": "2011-08-21 10:00:00",
            },
        ],
    }
    response = client.post("/predict/transactions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TX_CUST_99"
    assert "derived_features" in data
    # Should have 3 valid distinct invoices (cancellation dropped)
    assert data["derived_features"]["frequency"] == 3
    assert data["prediction"]["predicted_clv_90d"] >= 0.0
