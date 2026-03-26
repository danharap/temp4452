import argparse
from pathlib import Path

from src.baseline_model import (
    predict_probabilities,
    save_baseline_model,
    train_baseline_classifier,
)
from src.data_utils import load_split_dataframes, load_yaml_config
from src.eval_utils import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_misclassified_examples,
    save_predictions_csv,
    save_roc_curve_plot,
)
from src.features import extract_features_from_dataframe
from src.run_utils import create_run_dir, save_json, save_text, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train handcrafted-feature baseline model.")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml")
    return parser.parse_args()


def evaluate_split(model, df, image_size, threshold, run_dir, split_name, max_misclassified):
    X, y_true, paths = extract_features_from_dataframe(df, image_size=image_size)
    y_prob = predict_probabilities(model, X)
    metrics, y_pred = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    save_predictions_csv(paths, y_true, y_prob, y_pred, run_dir / f"predictions_{split_name}.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names=["real", "ai"],
        output_path=run_dir / f"confusion_matrix_{split_name}.png",
        title=f"Baseline Confusion Matrix ({split_name})",
    )
    if metrics.get("roc_auc") is not None:
        save_roc_curve_plot(y_true, y_prob, run_dir / f"roc_curve_{split_name}.png", f"Baseline ROC ({split_name})")

    save_misclassified_examples(
        image_paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=run_dir / f"misclassified_{split_name}",
        max_examples=max_misclassified,
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    run_dir = create_run_dir("baseline_train")
    save_yaml(cfg, run_dir / "config_used.yaml")

    train_df, val_df, test_df = load_split_dataframes(cfg["split_dir"])
    image_size = int(cfg.get("image_size", 224))

    X_train, y_train, _ = extract_features_from_dataframe(train_df, image_size=image_size)

    model = train_baseline_classifier(
        X_train,
        y_train,
        C=float(cfg["model"].get("C", 1.0)),
        max_iter=int(cfg["model"].get("max_iter", 3000)),
        seed=int(cfg.get("seed", 42)),
    )

    threshold = 0.5
    max_examples = int(cfg["output"].get("max_misclassified_examples", 30))

    val_metrics = evaluate_split(model, val_df, image_size, threshold, run_dir, "val", max_examples)
    test_metrics = evaluate_split(model, test_df, image_size, threshold, run_dir, "test", max_examples)

    model_out = Path(cfg["output"].get("model_path", "models/baseline_model.joblib"))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_baseline_model(model, model_out)
    save_baseline_model(model, run_dir / "baseline_model.joblib")

    merged = {"val": val_metrics, "test": test_metrics}
    save_json(merged, run_dir / "metrics.json")
    save_text(
        "\n".join([
            "Baseline training complete.",
            f"Saved main model: {model_out}",
            f"Test accuracy: {test_metrics['accuracy']:.4f}",
            f"Test F1: {test_metrics['f1']:.4f}",
            f"Test ROC-AUC: {test_metrics['roc_auc']}",
        ]),
        run_dir / "summary.txt",
    )

    print("Baseline training complete.")
    print(f"Model saved to: {model_out}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
