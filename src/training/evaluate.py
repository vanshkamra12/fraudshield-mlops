import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "pr_auc":    float(average_precision_score(y_true, y_scores)),
        "roc_auc":   float(roc_auc_score(y_true, y_scores)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-9)),
        "recall":    float(np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-9)),
        "threshold": threshold,
    }


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Threshold that maximises F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last element has no matching threshold
    return float(thresholds[best_idx])


def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str = "Precision-Recall Curve") -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#3F51B5", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (fraud rate {baseline:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Legit", "Pred: Fraud"])
    ax.set_yticklabels(["True: Legit", "True: Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 30) -> plt.Figure:
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([feature_names[i] for i in idx], importance[idx], color="#E91E63")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} Features")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
