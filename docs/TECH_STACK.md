# FraudShield — Tech Stack Decisions

Every tool was chosen deliberately. This document records WHY, not just WHAT.
In interviews, "why did you choose X over Y" is the question. This is the answer.

---

## Language: Python 3.11

Industry standard for ML. Best library ecosystem (PyTorch, scikit-learn, HuggingFace,
pandas). FastAPI is Python. Airflow is Python. Everything talks to everything.

---

## Streaming: Apache Kafka

Why over alternatives:
- **vs RabbitMQ:** Kafka retains messages (can replay). RabbitMQ deletes after consume.
  For fraud, replay is critical — if scoring service crashes, we don't lose transactions.
- **vs Redis Streams:** Kafka is the industry standard for high-throughput event pipelines.
  Any ML engineer knows Kafka. Redis Streams is a lightweight alternative, not a replacement.
- **vs direct API calls:** If scoring service is slow, direct calls block the sender.
  Kafka decouples producer from consumer — transactions queue up, nothing is lost.

---

## Feature Cache: Redis

Why over alternatives:
- **vs PostgreSQL for feature lookup:** Redis is in-memory, sub-millisecond reads.
  PostgreSQL queries take 5-50ms. We need features in <10ms to hit our 100ms SLA.
- **vs recomputing features every time:** Computing 30-day rolling averages on every
  transaction would require querying all historical data. Redis stores pre-computed values.

What we store: per-card transaction counts (1h/24h/7d), amount sums, mean spend, std spend.
Updated every time a new transaction is processed.

---

## Permanent Storage: PostgreSQL

Stores all predictions + ground truth labels for monitoring and retraining.
Relational structure suits our schema (transactions, model_metadata, drift_reports).

---

## Experiment Tracking: MLflow

Why over alternatives:
- **vs Weights & Biases:** W&B free tier has limits. MLflow is fully free, self-hosted,
  no account needed, no data leaves your machine.
- **vs manual logging:** MLflow gives experiment comparison UI, model versioning, artifact
  storage, model registry with staging/production states. Manual logging gives you a folder
  of .pkl files with no metadata.

What we track per run: parameters (model type, hyperparameters), metrics (PR-AUC, F1,
latency), artifacts (model file, feature importance plot, confusion matrix).

---

## Model Training: XGBoost + LightGBM + scikit-learn

- **Logistic Regression:** Interpretable baseline. If XGBoost barely beats this, the
  features are doing the heavy lifting, not model complexity.
- **XGBoost:** Industry gold standard for tabular fraud detection. Most Kaggle fraud
  competition winners use XGBoost or LightGBM.
- **LightGBM:** Faster than XGBoost on large datasets, often comparable accuracy.
  Important when retraining needs to be fast.

Why NOT deep learning: tree-based models consistently outperform neural nets on tabular
fraud data. Interpretability also matters — regulators ask "why was this blocked?"

Hyperparameter tuning: **Optuna** — Bayesian search, faster and smarter than GridSearchCV.

---

## Serving: FastAPI

- **vs Flask:** FastAPI is async, handles concurrent requests better, has automatic
  OpenAPI docs, built-in request validation via Pydantic.
- **vs Django:** Overkill for an API with 3 endpoints.

---

## Orchestration: Apache Airflow

Every ML team uses Airflow for batch pipelines. Our retraining DAG (data → features →
train → evaluate → register) is exactly what Airflow was built for. Directly transferable
skill to any ML job.

---

## Drift Detection: Evidently AI

Purpose-built for ML monitoring. One library call generates a full HTML report comparing
training data distribution vs recent predictions. Completely free and open-source.

Monitors: feature drift, prediction drift, data quality (missing values, out-of-range).

---

## Metrics + Dashboards: Prometheus + Grafana

- **Prometheus:** Scrapes metrics from FastAPI (latency histogram, request count, fraud rate).
- **Grafana:** Visualizes Prometheus data as live dashboards.

This is the exact stack every SRE and ML engineer uses at real companies.

---

## Containerization: Docker + Docker Compose

Docker packages each service with its exact dependencies — runs identically on any machine.

Why Compose over Kubernetes: Kubernetes manages containers across many machines (production
clusters). Compose runs multiple containers on one machine. For a laptop portfolio project,
Compose is exactly right. Kubernetes would be over-engineering.
