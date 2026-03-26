import argparse

import torch
from torch.utils.data import DataLoader

from src.baseline_model import load_baseline_model, predict_probabilities
from src.cnn_model import build_resnet_binary_classifier
from src.data_utils import load_split_dataframes, load_yaml_config
from src.dataset import ImageBinaryDataset, build_image_transform
from src.eval_utils import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_misclassified_examples,
    save_predictions_csv,
    save_roc_curve_plot,
)
from src.features import extract_features_from_dataframe
from src.robustness import make_degradation_config
from src.run_utils import create_run_dir, save_json, save_text, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model(s) on clean and degraded test data.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    return parser.parse_args()


def evaluate_baseline(model, df, image_size, threshold, degradation, run_dir, tag, max_examples):
    X, y_true, paths = extract_features_from_dataframe(df, image_size=image_size, degradation=degradation)
    y_prob = predict_probabilities(model, X)
    metrics, y_pred = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    save_predictions_csv(paths, y_true, y_prob, y_pred, run_dir / f"predictions_{tag}.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names=["real", "ai"],
        output_path=run_dir / f"confusion_matrix_{tag}.png",
        title=f"Baseline Confusion Matrix ({tag})",
    )
    if metrics.get("roc_auc") is not None:
        save_roc_curve_plot(y_true, y_prob, run_dir / f"roc_curve_{tag}.png", f"Baseline ROC ({tag})")

    save_misclassified_examples(paths, y_true, y_pred, run_dir / f"misclassified_{tag}", max_examples)
    return metrics


def evaluate_cnn(model, df, image_size, batch_size, num_workers, threshold, degradation, run_dir, tag, max_examples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = build_image_transform(image_size=image_size, is_train=False)
    ds = ImageBinaryDataset(df, transform=transform, return_path=True, degradation=degradation)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true, y_prob, paths = [], [], []
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            y_prob.extend(probs)
            y_true.extend(labels.numpy().tolist())
            paths.extend(batch_paths)

    metrics, y_pred = compute_classification_metrics(y_true, y_prob, threshold=threshold)

    save_predictions_csv(paths, y_true, y_prob, y_pred, run_dir / f"predictions_{tag}.csv")
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names=["real", "ai"],
        output_path=run_dir / f"confusion_matrix_{tag}.png",
        title=f"CNN Confusion Matrix ({tag})",
    )
    if metrics.get("roc_auc") is not None:
        save_roc_curve_plot(y_true, y_prob, run_dir / f"roc_curve_{tag}.png", f"CNN ROC ({tag})")

    save_misclassified_examples(paths, y_true, y_pred, run_dir / f"misclassified_{tag}", max_examples)
    return metrics


def build_eval_setups(eval_cfg):
    setups = []
    robustness_cfg = eval_cfg.get("robustness", {})

    if robustness_cfg.get("evaluate_clean", True):
        setups.append(("clean", None))

    for q in robustness_cfg.get("jpeg_qualities", []):
        setups.append((f"jpeg_q{q}", make_degradation_config("jpeg", quality=int(q))))

    for scale in robustness_cfg.get("resize_scales", []):
        tag = str(scale).replace(".", "p")
        setups.append((f"resize_{tag}", make_degradation_config("resize", scale=float(scale))))

    return setups


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    run_dir = create_run_dir("evaluate")
    save_yaml(cfg, run_dir / "config_used.yaml")

    _, _, test_df = load_split_dataframes(cfg["split_dir"])

    model_type = cfg.get("model_type", "cnn").lower()
    threshold = float(cfg.get("threshold", 0.5))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 2))
    max_examples = int(cfg["output"].get("max_misclassified_examples", 30))

    results = {}

    if model_type == "baseline":
        model = load_baseline_model(cfg["model_path"])
        for tag, degradation in build_eval_setups(cfg):
            results[tag] = evaluate_baseline(
                model=model,
                df=test_df,
                image_size=image_size,
                threshold=threshold,
                degradation=degradation,
                run_dir=run_dir,
                tag=tag,
                max_examples=max_examples,
            )
    elif model_type == "cnn":
        checkpoint = torch.load(cfg["model_path"], map_location="cpu")
        model = build_resnet_binary_classifier(
            backbone=checkpoint.get("backbone", "resnet18"),
            pretrained=False,
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["state_dict"])

        for tag, degradation in build_eval_setups(cfg):
            results[tag] = evaluate_cnn(
                model=model,
                df=test_df,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                threshold=threshold,
                degradation=degradation,
                run_dir=run_dir,
                tag=tag,
                max_examples=max_examples,
            )
    else:
        raise ValueError("model_type must be either 'baseline' or 'cnn'.")

    save_json(results, run_dir / "metrics.json")
    lines = ["Evaluation complete.", f"Model type: {model_type}", ""]
    for tag, metrics in results.items():
        lines.append(f"[{tag}] acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}, roc_auc={metrics.get('roc_auc')}")
    save_text("\n".join(lines), run_dir / "summary.txt")

    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()
