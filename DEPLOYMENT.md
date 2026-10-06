# Production Deployment Guide: Customer Lifetime Value (CLV) Prediction API

This guide provides end-to-end instructions for deploying the Customer Lifetime Value (CLV) prediction microservice in local and cloud environments.

---

## 1. Architecture Overview

The deployment wraps the Optuna-tuned Random Forest regression model into a production-ready **FastAPI** service:

```
                                  Client Request
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 Feature Payload               Raw Transactions
            (11 Pre-computed Features)         (Purchases/Items)
                         │                             │
                         │                     Feature Extraction
                         │                  (Clean + 11 RFM Features)
                         └──────────────┬──────────────┘
                                        ▼
                           Input Validation (Pydantic)
                                        ▼
                            Model Inference Pipeline
                     (RandomForest -> expm1 -> Monetary £)
                                        ▼
                        Tier & Decile Classification
                                        ▼
                           JSON Response with Actions
```

### Key Capabilities
- **Strict Leakage Prevention**: Features extracted from raw transactions adhere to cutoff boundaries.
- **Sub-15ms Latency**: Low-latency single customer scoring.
- **Batch Processing**: High-throughput `/predict/batch` for scoring thousands of customers.
- **Business Tiering**: Automatic assignment to tiers (VIP, High Growth, Moderate, Low, At-Risk) with actionable retention recommendations.

---

## 2. Local Setup & Testing

### Step 1: Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements-serve.txt
```

### Step 3: Run Tests
```bash
pytest tests/ -v
```

### Step 4: Launch API Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: **`http://localhost:8000/docs`**

---

## 3. API Endpoints Reference

### 1. Health Probe
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 11,
  "version": "1.0.0"
}
```

---

### 2. Predict Single Customer (`POST /predict`)
Scores a customer based on the 11 engineered features.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_14646",
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
    "has_single_purchase": 0
  }'
```
**Response:**
```json
{
  "customer_id": "CUST_14646",
  "predicted_clv_90d": 1284.60,
  "currency": "GBP",
  "value_tier": "Tier 1: VIP / High Value",
  "estimated_decile": 1,
  "actionable_recommendation": "Priority VIP: Assign dedicated retention budget, VIP perks, and high-touch support."
}
```

---

### 3. Predict from Raw Transactions (`POST /predict/transactions`)
Automatically performs cleaning (removes returns, filters non-merchandise fees), computes the 11 RFM/cadence/breadth features, and returns predicted 90-day CLV.

```bash
curl -X POST http://localhost:8000/predict/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "ONLINE_CUST_99",
    "transactions": [
      {
        "Invoice": "500001",
        "StockCode": "22423",
        "Description": "REGENCY CAKESTAND 3 TIER",
        "Quantity": 4,
        "Price": 12.75,
        "InvoiceDate": "2011-06-01 11:20:00"
      },
      {
        "Invoice": "500002",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 12,
        "Price": 2.55,
        "InvoiceDate": "2011-07-15 14:10:00"
      },
      {
        "Invoice": "500003",
        "StockCode": "47566",
        "Description": "PARTY BUNTING",
        "Quantity": 8,
        "Price": 4.95,
        "InvoiceDate": "2011-08-20 09:45:00"
      }
    ]
  }'
```

---

### 4. High-Throughput Batch Scoring (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST_01",
        "recency_days": 10.0,
        "frequency": 15,
        "monetary_total": 5200.0,
        "monetary_avg_order": 346.67,
        "tenure_days": 300.0,
        "avg_days_between_orders": 20.0,
        "active_months": 7,
        "average_monthly_spend": 527.78,
        "n_unique_products": 95,
        "avg_basket_size": 38.0,
        "has_single_purchase": 0
      },
      {
        "customer_id": "CUST_02",
        "recency_days": 210.0,
        "frequency": 1,
        "monetary_total": 45.0,
        "monetary_avg_order": 45.0,
        "tenure_days": 210.0,
        "avg_days_between_orders": 210.0,
        "active_months": 1,
        "average_monthly_spend": 6.52,
        "n_unique_products": 2,
        "avg_basket_size": 3.0,
        "has_single_purchase": 1
      }
    ]
  }'
```

---

## 4. Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t clv-prediction-api:latest .
```

### 2. Run the Container
```bash
docker run -d \
  --name clv-service \
  -p 8000:8000 \
  --restart unless-stopped \
  clv-prediction-api:latest
```

### 3. Or Use Docker Compose
```bash
docker compose up -d
```
Check container logs:
```bash
docker compose logs -f
```

---

## 5. Cloud Deployment Options

### Option 1: Google Cloud Run (Recommended for Serverless)
Cloud Run automatically scales to zero when idle and scales up on traffic spikes.

1. **Authenticate and configure project:**
   ```bash
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   ```

2. **Build and push image using Google Cloud Build:**
   ```bash
   gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/clv-prediction-api
   ```

3. **Deploy service:**
   ```bash
   gcloud run deploy clv-prediction-api \
     --image gcr.io/<YOUR_PROJECT_ID>/clv-prediction-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --cpu 1 \
     --min-instances 0 \
     --max-instances 10
   ```

---

### Option 2: Render / Railway
1. Push repository to GitHub.
2. Link repository in [Render](https://render.com) or [Railway](https://railway.app).
3. Select **Docker** environment.
4. Set port to `8000`.
5. Deploy automatically on commit.

---

### Option 3: AWS App Runner / ECS Fargate
1. **Push to Amazon ECR**:
   ```bash
   aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
   docker tag clv-prediction-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/clv-prediction-api:latest
   docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/clv-prediction-api:latest
   ```
2. **Deploy on AWS App Runner**: Select the ECR image, configure port `8000`, and start the service.

---

## 6. Production Maintenance & Monitoring

1. **Drift Detection**:
   - Monitor distribution shifts in `recency_days` and `average_monthly_spend` (the two strongest SHAP features).
   - Significant shifts indicate changing customer purchase cadences requiring retraining.

2. **Periodic Retraining Schedule**:
   - Schedule monthly batch retrains via `python -m src.train` using the latest 90-day observation window.
   - Run `python scripts/leakage_audit.py` in CI/CD before deploying any retrained artifact.
