"""
FraudShield daily monitoring DAG — runs Evidently drift check every day.

Schedule: daily at 06:00 UTC
DAG ID:   fraudshield_monitor
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "fraudshield",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_drift_check(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.monitoring.drift import run_drift_report

    summary = run_drift_report(
        hours=24,
        output_dir="/opt/airflow/docs/drift_reports",
    )

    print(f"Drift detected: {summary['drift_detected']}")
    print(f"Current window: {summary['n_current']:,} predictions")

    if summary.get("live_pr_auc"):
        print(f"Live PR-AUC: {summary['live_pr_auc']:.4f}")

    # flag for downstream alerting
    ctx["ti"].xcom_push(key="drift_detected", value=summary["drift_detected"])


def alert_on_drift(**ctx):
    drift = ctx["ti"].xcom_pull(task_ids="run_drift_check", key="drift_detected")

    if drift:
        print("ALERT: Data drift detected — consider triggering retraining.")
        # plug in Slack / PagerDuty / email here
    else:
        print("No drift detected — model distribution is stable.")


with DAG(
    dag_id="fraudshield_monitor",
    description="Daily Evidently drift check on live predictions",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraudshield", "monitoring"],
) as dag:

    t_drift = PythonOperator(
        task_id="run_drift_check",
        python_callable=run_drift_check,
    )

    t_alert = PythonOperator(
        task_id="alert_on_drift",
        python_callable=alert_on_drift,
    )

    t_drift >> t_alert
