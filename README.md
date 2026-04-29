# FraudShield — Real-time Fraud Detection

Built this as an end-to-end MLOps project to learn how fraud detection actually works in production — not just training a model and calling it done, but the full pipeline: streaming data, a live scoring API, automated retraining, and monitoring that tells you when the model starts going bad.

The dataset is IEEE-CIS (590k real transactions from Kaggle). The model flags fraud in under 10ms p95.

---

## How it works

Transactions come in through Kafka. A consumer picks them up, calls the FastAPI scoring service, gets a fraud score back, and logs everything to Postgres. Grafana shows what's happening live. Airflow handles weekly retraining. Evidently checks daily if the data distribution is shifting.

```
Transaction
    ↓
Kafka [transactions]
    ↓
Consumer → POST /score
    ↓
LightGBM (499 trees) → fraud_score 0–1
    ↓
threshold 0.74 → FRAUD / LEGIT
    ↓
Postgres + Kafka [predictions]
    ↓
Prometheus → Grafana dashboard
```

---

## Results

Trained on a strict time-based split (no random shuffle — day < 145 for train, rest for test). Numbers are honest.

| Metric | Score |
|--------|-------|
| PR-AUC | 0.516 |
| ROC-AUC | 0.907 |
| F1 | 0.504 |
| Precision | 0.547 |
| Recall | 0.468 |
| API p50 latency | 3.2ms |
| API p99 latency | 56ms |

PR-AUC of 0.516 sounds low but it's on a hard temporal split — no future data leaks into training. Kaggle leaderboard scores look better because most people use random splits. The ROC-AUC of 0.907 is more in line with what you'd expect.

---

## Stack

| What | Tool |
|------|------|
| Streaming | Kafka (KRaft, no Zookeeper) |
| Scoring API | FastAPI + LightGBM |
| Feature engineering | Custom sklearn-compatible pipeline |
| Experiment tracking | MLflow + MinIO (S3 artifacts) |
| Orchestration | Airflow 2.8 |
| Drift detection | Evidently AI |
| Metrics | Prometheus + Grafana |
| Storage | PostgreSQL (predictions + MLflow metadata) |
| Cache | Redis |
| Everything | Docker Compose (one command) |

---

## Project structure

```
fraudshield/
├── notebooks/
│   ├── 01_eda.ipynb              EDA on 590k transactions
│   ├── 02_feature_engineering.ipynb
│   └── 03_training.ipynb         two LightGBM runs, MLflow logged
├── src/
│   ├── features/
│   │   ├── pipeline.py           fit/transform/save/load
│   │   ├── encoders.py           smoothed target encoder
│   │   └── constants.py          feature lists, split day
│   ├── training/
│   │   ├── train.py              LightGBM + MLflow
│   │   └── evaluate.py           metrics, plots
│   ├── api/
│   │   ├── app.py                FastAPI, /predict /score /feedback /metrics
│   │   ├── models.py             SQLAlchemy ORM
│   │   └── schemas.py            Pydantic schemas
│   ├── ingestion/
│   │   ├── producer.py           streams test.parquet → Kafka
│   │   └── consumer.py           Kafka → /score → predictions topic
│   ├── monitoring/
│   │   └── drift.py              Evidently drift report on live predictions
│   └── orchestration/
│       ├── retrain_dag.py        weekly retraining DAG
│       └── monitor_dag.py        daily drift check DAG
├── infra/
│   ├── api/Dockerfile
│   ├── grafana/                  auto-provisioned dashboard + datasource
│   ├── prometheus/prometheus.yml
│   └── postgres/init.sql
├── docs/                         architecture diagram, EDA charts
├── models/                       lgbm_model.txt (committed), pipeline.joblib
├── docker-compose.yml            all 10 services
└── requirements.txt
```

---

## Running it

**Prerequisites:** Docker Desktop, Python 3.11+, the IEEE-CIS dataset from Kaggle.

Put the dataset files in `data/raw/ieee-fraud-detection/`:
```
train_transaction.csv
train_identity.csv
```

**Start everything:**
```bash
# core services first
docker-compose up -d postgres minio redis kafka

# wait ~20s, then
docker-compose up -d minio-init mlflow prometheus grafana

# wait ~30s, then build and start the API
docker-compose up -d --build api
```

**Run the feature pipeline + training (first time only):**
```bash
# generates data/processed/train.parquet, test.parquet, models/feature_pipeline.joblib
jupyter notebook notebooks/02_feature_engineering.ipynb

# trains LightGBM, logs to MLflow
jupyter notebook notebooks/03_training.ipynb
```

**Send transactions through the pipeline:**
```bash
# terminal 1 — consumer listens for transactions and calls the API
python3 -m src.ingestion.consumer

# terminal 2 — producer streams test data at 30 txn/s
python3 -m src.ingestion.producer --rate 30 --max 500
```

**Open the dashboards:**
| Service | URL | Login |
|---------|-----|-------|
| Grafana | http://localhost:3000 | admin / admin |
| MLflow | http://localhost:5001 | — |
| Airflow | http://localhost:8080 | admin / admin |
| API docs | http://localhost:8000/docs | — |

---

## Testing a single transaction

The API has an interactive docs page at http://localhost:8000/docs — just open it, click `/score`, and paste in some transaction data.

Or from the terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test_001",
    "amount": 5000.00,
    "card1": 1234.0, "card2": 5678.0, "card3": 9.0,
    "card4": "visa", "card5": 0.0, "card6": "debit",
    "addr1": 100.0, "addr2": 200.0, "dist1": 500.0,
    "ProductCD": "C",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "yahoo.com",
    "DeviceType": "mobile",
    "TransactionDT": 86400
  }'
```

Response:
```json
{
  "transaction_id": "test_001",
  "fraud_score": 0.231,
  "predicted_label": 0,
  "threshold": 0.7379
}
```

Score above 0.7379 → fraud flag. Below → legit.

---

## Feature engineering

434 raw columns → 113 features. Main things the pipeline does:

- drops columns with >80% missing values
- `amt_log` = log1p(amount), `amt_zscore` = how unusual is this amount for this specific card
- hour of day + day of week from the raw timestamp
- smoothed target encoding for card brand, email domain, product code
- M-flags (T/F/NaN) converted to 1/0/-1
- top 50 V-features by correlation with fraud label
- median imputation for numeric features, using training-set medians only

Everything is fit on train, applied to test — no leakage.

---

## Retraining

Two Airflow DAGs in `src/orchestration/`:

`fraudshield_retrain` runs every Sunday at 02:00 UTC:
1. checks if there are 5000+ new labelled predictions since last run
2. re-fits the feature pipeline on fresh data
3. trains a new LightGBM model, logs to MLflow
4. compares new PR-AUC vs current champion in registry
5. promotes to Production only if the challenger wins

`fraudshield_monitor` runs daily at 06:00 UTC:
1. pulls last 24h of predictions from Postgres
2. runs Evidently drift report vs training reference
3. logs live PR-AUC on labelled predictions
4. alerts if distribution has shifted

---

## What I'd add with more time

- proper unit tests (pytest)
- Redis feature cache so the pipeline doesn't recompute on every request
- more aggressive feature engineering — card-level aggregates, velocity features
- a proper CI/CD pipeline (GitHub Actions → Docker build → deploy)
- load testing to find actual throughput ceiling
