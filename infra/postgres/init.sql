-- Runs once when the PostgreSQL container starts for the first time.
-- Creates the extra databases MLflow and Airflow need.
-- The main 'fraudshield' database is already created by POSTGRES_DB env var.

CREATE DATABASE mlflow;
CREATE DATABASE airflow;

-- Grant the app user access to all three
GRANT ALL PRIVILEGES ON DATABASE mlflow TO fraudshield;
GRANT ALL PRIVILEGES ON DATABASE airflow TO fraudshield;
GRANT ALL PRIVILEGES ON DATABASE fraudshield TO fraudshield;

-- Connect to fraudshield database and create tables
\connect fraudshield

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL UNIQUE,
    fraud_score FLOAT NOT NULL,
    predicted_label INT NOT NULL,
    actual_label INT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
