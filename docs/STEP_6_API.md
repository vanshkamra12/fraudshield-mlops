# Step 6: FastAPI Scoring Service

Real-time fraud detection scoring API. Accepts transaction features, returns fraud probability and predicted label. All predictions logged to PostgreSQL.

## Running the Service

### With Docker

```bash
docker-compose up api
```

Service starts on `http://localhost:8000`

### Without Docker (local development)

```bash
pip install -r requirements.txt
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "pipeline_loaded": true
}
```

### Predict Fraud Score

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_12345",
    "amount": 150.50,
    "card1": 1234.0,
    "card2": 5678.0,
    "card3": 9.0,
    "card4": "visa",
    "card5": 0.0,
    "card6": "american_express",
    "addr1": 100.0,
    "addr2": 200.0,
    "dist1": 50.0,
    "ProductCD": "C",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "yahoo.com",
    "DeviceType": "desktop",
    "TransactionDT": 86400,
    "C1": 0.1,
    "C2": 0.2,
    "D1": 1.0,
    "M1": "T",
    "V45": 0.5
  }'
```

Response:
```json
{
  "transaction_id": "txn_12345",
  "fraud_score": 0.234,
  "predicted_label": 0,
  "threshold": 0.7379
}
```

**Fields:**
- `fraud_score`: Raw fraud probability (0–1). Higher = more likely fraud
- `predicted_label`: 0 = legitimate, 1 = fraud (based on threshold)
- `threshold`: Decision boundary (default 0.7379 from tuned model validation)

### Record Ground Truth Feedback

Once the actual outcome is known (e.g., transaction confirmed as fraud or legitimate), send feedback:

```bash
curl -X POST "http://localhost:8000/feedback?transaction_id=txn_12345&actual_label=0"
```

Response:
```json
{
  "status": "feedback recorded"
}
```

This updates the `predictions` table with `actual_label`, enabling:
- Model performance monitoring
- Drift detection
- Retraining triggers

## Database Schema

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    fraud_score FLOAT NOT NULL,
    predicted_label INT NOT NULL,
    actual_label INT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

**Columns:**
- `id`: Unique prediction ID
- `transaction_id`: External transaction identifier
- `fraud_score`: Model output (0–1)
- `predicted_label`: Binarized prediction (0 or 1)
- `actual_label`: Ground truth label (NULL until feedback received)
- `created_at`: Timestamp when prediction was made

## Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/fraudshield
FEATURE_PIPELINE_PATH=models/feature_pipeline.joblib
MODEL_PATH=models/lgbm_model.txt
```

## Model / Pipeline Loading

On startup, the API:
1. Loads `FraudFeaturePipeline` from disk (transforms raw features → 113 engineered features)
2. Loads LightGBM booster model (499 trees, trained on time-based split)
3. Creates PostgreSQL `predictions` table if not exists
4. Reports health status

If model/pipeline fails to load, `/health` returns `model_loaded: false` and `/predict` returns 503.

## Feature Requirements

The `/predict` endpoint accepts the following features. Send `null` for optional fields:

**Required:**
- `transaction_id`: str (unique)
- `amount`: float
- `card1`–`card6`: float + str (card attributes)
- `addr1`, `addr2`, `dist1`: float (address features)
- `ProductCD`: str
- `P_emaildomain`, `R_emaildomain`: str (email domains)
- `DeviceType`: str (mobile/desktop)
- `TransactionDT`: int (seconds since epoch)

**Optional (50+ V-features, C-features, D-features, M-flags, identity features)**

All features from the training notebook schema are accepted. Missing numeric features default to training-set medians (filled by pipeline).

## Monitoring & Next Steps

- Step 7: Kafka producer/consumer for streaming predictions
- Step 8: Airflow DAG to retrain models daily
- Step 9: Evidently AI drift detection + Grafana dashboards (monitor `predictions` table)
