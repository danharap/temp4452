import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = project_root / "runs"

    eval_runs = sorted(runs_dir.glob("evaluate_*"))
    if not eval_runs:
        raise FileNotFoundError("No evaluate_* run folders found in runs/.")

    eval_run = eval_runs[-1]
    metrics_path = eval_run / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {eval_run}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = []
    ordered_tags = ["clean", "jpeg_q85", "jpeg_q60", "jpeg_q40", "resize_0p75", "resize_0p5"]
    tags = [t for t in ordered_tags if t in metrics] + [t for t in metrics.keys() if t not in ordered_tags]

    for tag in tags:
        m = metrics[tag]
        rows.append(
            [
                tag,
                f"{m.get('accuracy', 0.0):.4f}",
                f"{m.get('precision', 0.0):.4f}",
                f"{m.get('recall', 0.0):.4f}",
                f"{m.get('f1', 0.0):.4f}",
                f"{m.get('roc_auc', 0.0):.4f}" if m.get("roc_auc") is not None else "n/a",
            ]
        )

    col_labels = ["Condition", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig_h = max(3.0, 0.55 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax.set_title("Evaluate Metrics Summary", fontsize=14, pad=14)

    output_path = eval_run / "evaluate_metrics_table.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
