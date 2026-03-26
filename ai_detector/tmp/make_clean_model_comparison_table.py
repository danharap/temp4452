import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_run(runs_dir: Path, prefix: str) -> Path:
    candidates = sorted(runs_dir.glob(f"{prefix}_*"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* run folders found in {runs_dir}")
    return candidates[-1]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = project_root / "runs"

    baseline_run = _latest_run(runs_dir, "baseline_train")
    cnn_run = _latest_run(runs_dir, "cnn_train")

    baseline_metrics = _load_json(baseline_run / "metrics.json")
    cnn_metrics = _load_json(cnn_run / "metrics.json")

    baseline_test = baseline_metrics["test"]
    cnn_test = cnn_metrics["test_metrics"]

    rows = [
        [
            "Baseline test",
            f"{baseline_test.get('accuracy', 0.0):.4f}",
            f"{baseline_test.get('precision', 0.0):.4f}",
            f"{baseline_test.get('recall', 0.0):.4f}",
            f"{baseline_test.get('f1', 0.0):.4f}",
            f"{baseline_test.get('roc_auc', 0.0):.4f}" if baseline_test.get("roc_auc") is not None else "n/a",
        ],
        [
            "CNN test",
            f"{cnn_test.get('accuracy', 0.0):.4f}",
            f"{cnn_test.get('precision', 0.0):.4f}",
            f"{cnn_test.get('recall', 0.0):.4f}",
            f"{cnn_test.get('f1', 0.0):.4f}",
            f"{cnn_test.get('roc_auc', 0.0):.4f}" if cnn_test.get("roc_auc") is not None else "n/a",
        ],
    ]

    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("Clean Model Comparison (Test Set)", fontsize=14, pad=12)

    out_path = runs_dir / "clean_model_comparison_table.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
