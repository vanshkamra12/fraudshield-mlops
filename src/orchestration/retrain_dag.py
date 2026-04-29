"""
FraudShield retraining DAG — runs weekly, retrains LightGBM, promotes if better.

Schedule: every Sunday at 02:00 UTC
DAG ID:   fraudshield_retrain

Tasks:
  1. check_data_volume    — fail fast if not enough new predictions in DB
  2. run_feature_pipeline — fit pipeline on train split, transform both sets
  3. train_model          — LightGBM + MLflow run, returns run_id + metrics
  4. evaluate_model       — compare new PR-AUC vs current champion
  5. promote_model        — register as FraudShield vN if new model wins
  6. notify               — log summary (extend to Slack/email as needed)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ─────────────────────────────────────────
# DAG defaults
# ─────────────────────────────────────────
DEFAULT_ARGS = {
    "owner": "fraudshield",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

DATA_DIR   = "/opt/airflow/data/processed"
MODEL_DIR  = "/opt/airflow/models"
MIN_ROWS   = 5_000   # minimum new predictions needed to trigger retrain
PR_AUC_MIN = 0.48    # only promote if new model beats this floor


# ─────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────

def check_data_volume(**ctx):
    """Verify enough new labelled predictions exist since last run."""
    import psycopg2
    import os

    db_url = os.getenv("DATABASE_URL", "postgresql://fraudshield:fraudshield123@postgres:5432/fraudshield")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    last_run = ctx["prev_execution_date"] or datetime(2000, 1, 1)
    cur.execute(
        "SELECT COUNT(*) FROM predictions WHERE actual_label IS NOT NULL AND created_at > %s",
        (last_run,)
    )
    count = cur.fetchone()[0]
    conn.close()

    print(f"New labelled predictions since {last_run}: {count:,}")

    if count < MIN_ROWS:
        raise ValueError(
            f"Only {count} new labelled rows — need {MIN_ROWS} to retrain. Skipping."
        )

    ctx["ti"].xcom_push(key="new_rows", value=count)


def run_feature_pipeline(**ctx):
    """Re-fit feature pipeline on train split, save updated artefacts."""
    import sys
    import os
    sys.path.insert(0, "/opt/airflow")

    import pandas as pd
    from src.features.pipeline import FraudFeaturePipeline
    from src.features.constants import TRAIN_TEST_SPLIT_DAY

    raw_dir = "/opt/airflow/data/raw/ieee-fraud-detection"
    trans = pd.read_csv(f"{raw_dir}/train_transaction.csv")
    ident = pd.read_csv(f"{raw_dir}/train_identity.csv")
    df = trans.merge(ident, on="TransactionID", how="left")

    df["day"] = df["TransactionDT"] // 86400
    train_df = df[df["day"] < TRAIN_TEST_SPLIT_DAY].drop(columns="day").reset_index(drop=True)
    test_df  = df[df["day"] >= TRAIN_TEST_SPLIT_DAY].drop(columns="day").reset_index(drop=True)

    y_train = train_df.pop("isFraud")
    y_test  = test_df.pop("isFraud")

    pipeline = FraudFeaturePipeline()
    X_train = pipeline.fit_transform(train_df, y_train)
    X_test  = pipeline.transform(test_df)

    os.makedirs(DATA_DIR, exist_ok=True)
    train_out = X_train.copy(); train_out["isFraud"] = y_train.values
    test_out  = X_test.copy();  test_out["isFraud"]  = y_test.values

    train_out.to_parquet(f"{DATA_DIR}/train.parquet", index=False)
    test_out.to_parquet(f"{DATA_DIR}/test.parquet", index=False)
    pipeline.save(f"{MODEL_DIR}/feature_pipeline.joblib")

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {X_train.shape[1]}")


def train_model(**ctx):
    """Run LightGBM training and log to MLflow."""
    import sys
    import os
    sys.path.insert(0, "/opt/airflow")

    from src.training.train import train

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    run_ts = datetime.utcnow().strftime("%Y%m%d_%H%M")

    results = train(
        data_dir=DATA_DIR,
        model_dir=MODEL_DIR,
        mlflow_uri=mlflow_uri,
        experiment_name="fraudshield",
        run_name=f"retrain_{run_ts}",
        register_model=False,   # only register after evaluation gate
    )

    print(f"Run ID: {results['run_id']}")
    print(f"Test PR-AUC: {results['test']['pr_auc']:.4f}")

    ctx["ti"].xcom_push(key="run_id",   value=results["run_id"])
    ctx["ti"].xcom_push(key="pr_auc",   value=results["test"]["pr_auc"])
    ctx["ti"].xcom_push(key="threshold", value=results["threshold"])


def evaluate_model(**ctx):
    """Compare new model PR-AUC vs current champion. Branch on result."""
    ti = ctx["ti"]
    new_pr_auc = ti.xcom_pull(task_ids="train_model", key="pr_auc")

    import mlflow
    import os
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    # get current champion PR-AUC from registry
    champion_pr_auc = 0.0
    try:
        versions = client.get_latest_versions("FraudShield", stages=["Production"])
        if versions:
            champ_run = client.get_run(versions[0].run_id)
            champion_pr_auc = champ_run.data.metrics.get("test_pr_auc", 0.0)
    except Exception:
        pass

    print(f"Champion PR-AUC : {champion_pr_auc:.4f}")
    print(f"Challenger PR-AUC: {new_pr_auc:.4f}")
    print(f"Floor           : {PR_AUC_MIN:.4f}")

    ti.xcom_push(key="champion_pr_auc", value=champion_pr_auc)

    if new_pr_auc > champion_pr_auc and new_pr_auc >= PR_AUC_MIN:
        print("✓ Challenger wins — will promote")
        return "promote_model"
    else:
        print("✗ Champion holds — skipping promotion")
        return "skip_promotion"


def promote_model(**ctx):
    """Register winning model as FraudShield vN in MLflow registry."""
    import mlflow
    import os
    import shutil

    ti = ctx["ti"]
    run_id    = ti.xcom_pull(task_ids="train_model", key="run_id")
    threshold = ti.xcom_pull(task_ids="train_model", key="threshold")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, "FraudShield")
    print(f"Registered FraudShield v{mv.version}")

    client.transition_model_version_stage(
        name="FraudShield",
        version=mv.version,
        stage="Production",
    )

    # update the threshold file the API reads at startup
    threshold_path = f"{MODEL_DIR}/threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    # copy fresh booster to model dir so API picks it up on next restart
    src_txt = f"{MODEL_DIR}/lgbm_model.txt"
    if os.path.exists(src_txt):
        shutil.copy(src_txt, f"{MODEL_DIR}/lgbm_model_prev.txt")

    print(f"Promoted to Production. Threshold: {threshold:.4f}")


def notify(**ctx):
    """Log a run summary (replace print with Slack/email webhook as needed)."""
    ti = ctx["ti"]
    new_pr_auc      = ti.xcom_pull(task_ids="train_model",  key="pr_auc")
    champion_pr_auc = ti.xcom_pull(task_ids="evaluate_model", key="champion_pr_auc")
    run_id          = ti.xcom_pull(task_ids="train_model",  key="run_id")

    promoted = new_pr_auc > (champion_pr_auc or 0.0)

    print("=" * 50)
    print("FRAUDSHIELD RETRAINING SUMMARY")
    print("=" * 50)
    print(f"Run ID          : {run_id}")
    print(f"New PR-AUC      : {new_pr_auc:.4f}")
    print(f"Champion PR-AUC : {champion_pr_auc:.4f}")
    print(f"Promoted        : {'YES' if promoted else 'NO'}")
    print("=" * 50)


# ─────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────
with DAG(
    dag_id="fraudshield_retrain",
    description="Weekly LightGBM retraining + champion/challenger evaluation",
    default_args=DEFAULT_ARGS,
    schedule="0 2 * * 0",    # every Sunday at 02:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraudshield", "retraining"],
) as dag:

    t_check = PythonOperator(
        task_id="check_data_volume",
        python_callable=check_data_volume,
    )

    t_pipeline = PythonOperator(
        task_id="run_feature_pipeline",
        python_callable=run_feature_pipeline,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_evaluate = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_skip = EmptyOperator(task_id="skip_promotion")

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
        trigger_rule="none_failed_min_one_success",
    )

    # ── Pipeline ─────────────────────────
    t_check >> t_pipeline >> t_train >> t_evaluate >> [t_promote, t_skip] >> t_notify
