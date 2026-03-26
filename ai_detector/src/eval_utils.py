import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics, y_pred.tolist()


def save_confusion_matrix_plot(conf_matrix, class_names, output_path: str | Path, title: str):
    cm = np.asarray(conf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_prob, output_path: str | Path, title: str):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions_csv(paths, y_true, y_prob, y_pred, output_path: str | Path):
    rows = []
    for p, yt, yp, yhat in zip(paths, y_true, y_prob, y_pred):
        rows.append(
            {
                "path": p,
                "true_label": int(yt),
                "pred_label": int(yhat),
                "prob_ai": float(yp),
                "true_label_name": "ai" if int(yt) == 1 else "real",
                "pred_label_name": "ai" if int(yhat) == 1 else "real",
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_misclassified_examples(
    image_paths,
    y_true,
    y_pred,
    output_dir: str | Path,
    max_examples: int = 20,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for path, yt, yp in zip(image_paths, y_true, y_pred):
        if int(yt) == int(yp):
            continue

        src = Path(path)
        if not src.exists():
            continue

        name = f"true_{'ai' if int(yt)==1 else 'real'}_pred_{'ai' if int(yp)==1 else 'real'}__{src.name}"
        dst = output_dir / name
        try:
            shutil.copy2(src, dst)
            saved += 1
        except OSError:
            continue

        if saved >= max_examples:
            break
