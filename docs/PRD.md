# FraudShield — Product Requirements Document

**Version:** 1.0  
**Date:** 2026-04-22  
**Author:** Vansh Kamra  
**Status:** Approved — In Development

---

## 1. Problem Statement

Every second, thousands of credit/debit card transactions happen globally. A small percentage
of these are fraudulent — a stolen card being used, a fake merchant, an account takeover.

Banks need to decide in under 100 milliseconds whether to allow or block a transaction.
Too slow → customer experience is ruined. Wrong decision → either fraud goes through (bank
loses money) or a legitimate transaction is blocked (customer is angry).

Currently, most tutorials and student projects train a model and stop there. That model
sitting in a Jupyter notebook is useless in production. This project builds the **complete
system** — not just the model, but the pipeline that feeds it, serves it, and maintains it.

---

## 2. What We Are Building

A real-time financial fraud detection system with the following capabilities:

- Ingest a continuous stream of financial transactions
- Score each transaction for fraud probability within 100ms
- Serve scoring decisions via a REST API
- Track all ML experiments and model versions
- Monitor model health and data drift over time
- Automatically trigger model retraining when performance degrades

---

## 3. Users of This System

| User | What they need |
|------|---------------|
| Payment gateway | Call /predict API, get fraud score, decide to allow/block |
| ML engineer (us) | Train models, track experiments, deploy new versions |
| Risk analyst | View dashboards, understand fraud patterns, tune thresholds |
| System operator | Monitor system health, latency, throughput |

---

## 4. Functional Requirements

### 4.1 Real-time Scoring (Core)
- System must score a transaction and return a decision in < 100ms (p99)
- Input: raw transaction fields (card_id, amount, merchant_type, timestamp, location, etc.)
- Output: fraud_probability (0.0 to 1.0), decision (ALLOW/BLOCK), model_version

### 4.2 Data Pipeline
- Transactions arrive as a stream (simulated via Kafka producer)
- Feature engineering must run on each transaction before scoring
- User's historical spend context must be fetched from Redis (fast cache)

### 4.3 Model Training
- Support training on the IEEE-CIS Fraud Detection dataset (Kaggle, free)
- Compare at minimum 3 model types: Logistic Regression, XGBoost, LightGBM
- All experiments tracked in MLflow (parameters, metrics, artifacts)
- Best model promoted to "Production" in MLflow Model Registry

### 4.4 Monitoring
- Track data drift weekly using Evidently AI (are new transactions different from training?)
- Track model performance over time (precision, recall, PR-AUC on sliding window)
- Expose system metrics (latency, throughput, fraud_rate) via Prometheus → Grafana
- Auto-trigger retraining via Airflow DAG when drift score exceeds threshold

### 4.5 Model Retraining Pipeline
- Airflow DAG: pull new data → feature engineering → train → evaluate → promote if better

---

## 5. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Prediction latency | p99 < 100ms |
| Throughput | > 500 transactions/sec |
| Model PR-AUC | > 0.85 on test set |
| System uptime | Services restart automatically on failure (Docker restart policy) |
| Cost | $0 — entire stack runs locally via Docker |

---

## 6. Out of Scope (What We Are NOT Building)

- A frontend UI / mobile app
- Real bank integration (we simulate transactions)
- Real-time retraining (we retrain on a schedule / drift trigger, not per transaction)
- Authentication/authorization for the API (not needed for portfolio project)
- Multi-region deployment

---

## 7. Dataset

**Source:** IEEE-CIS Fraud Detection (Kaggle)  
**Link:** https://www.kaggle.com/c/ieee-fraud-detection/data  
**Size:** ~590k transactions, 400+ features across 2 tables  
**Fraud rate:** ~3.5% (heavily imbalanced — this is a feature, not a bug, mirrors reality)  
**License:** Free for non-commercial/educational use  

Why this dataset: it has realistic complexity (joins, missing values, high cardinality),
mirrors what real fraud teams work with, and is an industry-recognized benchmark.

---

## 8. Success Metrics

The project is considered complete when:

- [ ] `docker-compose up` starts all 8 services without errors
- [ ] A simulated transaction stream flows end-to-end: producer → Kafka → feature service → scoring → PostgreSQL
- [ ] MLflow UI shows at least 10 tracked experiments across 3+ model types
- [ ] FastAPI /predict endpoint returns a response in < 100ms for single transactions
- [ ] Grafana dashboard shows live fraud rate, model latency, transaction throughput
- [ ] Evidently generates a drift report comparing train distribution vs recent transactions
- [ ] Airflow DAG runs successfully end-to-end (data → train → evaluate → register)
- [ ] README contains architecture diagram, setup instructions, and performance numbers

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Kafka complexity for first-timer | Use simplest config first, add partitions/replicas later |
| Dataset too large for local machine | Use 20% sample for development, full dataset for final training |
| Docker resource usage heavy (8 services) | Set memory limits in docker-compose, use Apple Silicon optimized images |
| Class imbalance kills model performance | Explicitly compare SMOTE vs class_weight vs neither in MLflow |
