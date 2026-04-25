"""
Main training script — LightGBM + MLflow.

Usage (from project root):
    python -m src.training.train

Or import and call train() from the notebook.
"""
import os
import tempfile

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.training.evaluate import (
    compute_metrics,
    find_best_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
)

# ──────────────────────────────────────────
# Default hyperparameters
# ──────────────────────────────────────────
DEFAULT_PARAMS = {
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 100,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "lambda_l1":         0.1,
    "lambda_l2":         0.1,
    "metric":            "auc",
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


def train(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    mlflow_uri: str | None = None,
    experiment_name: str = "fraudshield",
    run_name: str | None = None,
    params: dict | None = None,
    register_model: bool = True,
) -> dict:
    """
    Train LightGBM, log everything to MLflow, return test metrics.
    """
    # ── MLflow setup ──────────────────────
    tracking_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # When using the docker MLflow server, artifacts go to MinIO.
    # Set boto3 env vars so the client can upload to MinIO directly.
    if tracking_uri.startswith("http"):
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")

    # ── Load data ─────────────────────────
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    test_df  = pd.read_parquet(f"{data_dir}/test.parquet")

    y_train = train_df.pop("isFraud").values.astype(np.int8)
    y_test  = test_df.pop("isFraud").values.astype(np.int8)
    X_train = train_df.values.astype(np.float32)
    X_test  = test_df.values.astype(np.float32)
    feature_names = train_df.columns.tolist()

    # ── Validation split (temporal — last 10% of train) ────
    val_size = int(len(X_train) * 0.1)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr,  y_tr  = X_train[:-val_size],  y_train[:-val_size]

    # class imbalance weight
    scale_pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    hp = {**DEFAULT_PARAMS, **(params or {})}
    hp["scale_pos_weight"] = scale_pos_weight

    print(f"Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # ── Train ─────────────────────────────
    with mlflow.start_run(run_name=run_name) as run:

        model = lgb.LGBMClassifier(**hp)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        best_iter = model.best_iteration_
        print(f"\nBest iteration: {best_iter}")

        # ── Scores ────────────────────────
        val_scores  = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]

        # Tune threshold on val, evaluate on test
        threshold = find_best_threshold(y_val, val_scores)
        val_metrics  = compute_metrics(y_val,  val_scores,  threshold)
        test_metrics = compute_metrics(y_test, test_scores, threshold)

        print(f"\nVal  PR-AUC: {val_metrics['pr_auc']:.4f}  F1: {val_metrics['f1']:.4f}")
        print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}  F1: {test_metrics['f1']:.4f}")

        # ── Log to MLflow ─────────────────
        mlflow.log_params({**hp, "best_iteration": best_iter})

        for split, m in [("val", val_metrics), ("test", test_metrics)]:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})

        # ── Figures ───────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            pr_fig = plot_pr_curve(y_test, test_scores, "Test PR Curve")
            pr_path = f"{tmp}/pr_curve.png"
            pr_fig.savefig(pr_path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(pr_path, "plots")

            cm_fig = plot_confusion_matrix(y_test, (test_scores >= threshold).astype(int))
            cm_path = f"{tmp}/confusion_matrix.png"
            cm_fig.savefig(cm_path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(cm_path, "plots")

            fi_fig = plot_feature_importance(model, feature_names, top_n=30)
            fi_path = f"{tmp}/feature_importance.png"
            fi_fig.savefig(fi_path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(fi_path, "plots")

        import matplotlib.pyplot as plt
        plt.close("all")

        # ── Log model ─────────────────────
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="FraudShield" if register_model else None,
        )

        # ── Also save locally ─────────────
        os.makedirs(model_dir, exist_ok=True)
        model.booster_.save_model(f"{model_dir}/lgbm_model.txt")

        run_id = run.info.run_id
        print(f"\nMLflow run: {run_id}")
        print(f"Tracking UI: {tracking_uri}")

    return {"run_id": run_id, "val": val_metrics, "test": test_metrics, "threshold": threshold}


if __name__ == "__main__":
    results = train()
    print("\nDone. Test metrics:")
    for k, v in results["test"].items():
        print(f"  {k}: {v:.4f}")
