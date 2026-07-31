from __future__ import annotations
import json
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, matthews_corrcoef, brier_score_loss 


# picking decision threshold that maximizes f-beta on val data
# here FN (missed failure) is far more expensive than a false alarm (FP) in maintenance settings.
def best_pick_by_threshold_by_fbeta(y_true, y_prob, beta: float = 2.0) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1] 
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    fbeta = np.nan_to_num(fbeta)
    if len(fbeta) == 0:
        return 0.5
    
    return float(thresholds[int(np.argmax(fbeta))])


def classification_metrics(y_true,y_pred,y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    return metrics
    
# y_true_binary: ground-truth failure-window label used only for offline validation of the unsupervised detector (not used in training).
def anamoly_metrics(y_true_binary, anomaly_flags, scores) -> dict:
    metrics = classification_metrics(y_true_binary, anomaly_flags,scores / (scores.max() + 1e-9))
    metrics["mean_reconstruction_error"] = float(np.mean(scores))
    return metrics

def plot_roc_curve(y_true, y_prob, out_path: str):
    fpr, tpr, _= roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    

def plot_pr_curve(y_true, y_proba, out_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Healthy", "Failure-soon"])
    plt.yticks([0, 1], ["Healthy", "Failure-soon"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_metrics_json(metrics: dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
        