import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

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
from src.run_utils import create_run_dir, save_json, save_text, save_yaml
from src.train_utils import evaluate_model_logits, save_history_csv, set_seed, train_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN (transfer-learned ResNet) for real-vs-AI detection.")
    parser.add_argument("--config", type=str, default="configs/cnn_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    run_dir = create_run_dir("cnn_train")
    save_yaml(cfg, run_dir / "config_used.yaml")

    train_df, val_df, test_df = load_split_dataframes(cfg["split_dir"])

    image_size = int(cfg.get("image_size", 224))
    train_transform = build_image_transform(image_size=image_size, is_train=True)
    eval_transform = build_image_transform(image_size=image_size, is_train=False)

    train_ds = ImageBinaryDataset(train_df, transform=train_transform)
    val_ds = ImageBinaryDataset(val_df, transform=eval_transform)
    test_ds = ImageBinaryDataset(test_df, transform=eval_transform, return_path=True)

    train_cfg = cfg["train"]
    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet_binary_classifier(
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    model, history, best_val_metrics = train_cnn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(train_cfg.get("epochs", 8)),
        threshold=float(train_cfg.get("threshold", 0.5)),
    )

    save_history_csv(history, run_dir / "train_history.csv")

    test_loss, y_true, y_prob, paths = evaluate_model_logits(model, test_loader, criterion, device)
    test_metrics, y_pred = compute_classification_metrics(y_true, y_prob, threshold=float(train_cfg.get("threshold", 0.5)))
    test_metrics["loss"] = float(test_loss)

    save_predictions_csv(paths, y_true, y_prob, y_pred, run_dir / "predictions_test.csv")
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        class_names=["real", "ai"],
        output_path=run_dir / "confusion_matrix.png",
        title="CNN Confusion Matrix (test)",
    )
    if test_metrics.get("roc_auc") is not None:
        save_roc_curve_plot(y_true, y_prob, run_dir / "roc_curve.png", "CNN ROC (test)")

    save_misclassified_examples(
        image_paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=run_dir / "misclassified_examples",
        max_examples=int(cfg["output"].get("max_misclassified_examples", 30)),
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "backbone": cfg["model"].get("backbone", "resnet18"),
        "image_size": image_size,
        "threshold": float(train_cfg.get("threshold", 0.5)),
        "class_names": ["real", "ai"],
        "best_val_metrics": best_val_metrics,
    }

    model_out = Path(cfg["output"].get("model_path", "models/cnn_model.pt"))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)
    torch.save(checkpoint, run_dir / "model.pt")

    report = {"best_val_metrics": best_val_metrics, "test_metrics": test_metrics}
    save_json(report, run_dir / "metrics.json")
    save_text(
        "\n".join([
            "CNN training complete.",
            f"Device: {device}",
            f"Saved main model: {model_out}",
            f"Test accuracy: {test_metrics['accuracy']:.4f}",
            f"Test F1: {test_metrics['f1']:.4f}",
            f"Test ROC-AUC: {test_metrics['roc_auc']}",
        ]),
        run_dir / "summary.txt",
    )

    print("CNN training complete.")
    print(f"Model saved to: {model_out}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
