-- Runs once when the PostgreSQL container starts for the first time.
-- Creates the extra databases MLflow and Airflow need.
-- The main 'fraudshield' database is already created by POSTGRES_DB env var.

CREATE DATABASE mlflow;
CREATE DATABASE airflow;

-- Grant the app user access to all three
GRANT ALL PRIVILEGES ON DATABASE mlflow TO fraudshield;
GRANT ALL PRIVILEGES ON DATABASE airflow TO fraudshield;
GRANT ALL PRIVILEGES ON DATABASE fraudshield TO fraudshield;
