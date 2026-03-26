import argparse

import torch

from src.cnn_model import build_resnet_binary_classifier
from src.data_utils import load_yaml_config
from src.dataset import build_image_transform
from src.predict_utils import collect_input_images, predict_images_with_cnn
from src.run_utils import create_run_dir, save_predictions_table, save_text, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict real vs AI-generated for one image or folder.")
    parser.add_argument("--config", type=str, default="configs/predict_config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to image or folder")
    parser.add_argument("--model", type=str, default=None, help="Optional model checkpoint path override")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    run_dir = create_run_dir("predict")
    save_yaml(cfg, run_dir / "config_used.yaml")

    model_path = args.model if args.model else cfg["model_path"]
    checkpoint = torch.load(model_path, map_location="cpu")

    model = build_resnet_binary_classifier(
        backbone=checkpoint.get("backbone", "resnet18"),
        pretrained=False,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"])

    image_size = int(checkpoint.get("image_size", cfg.get("image_size", 224)))
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", cfg.get("threshold", 0.5)))

    transform = build_image_transform(image_size=image_size, is_train=False)
    image_paths = collect_input_images(args.input)

    predictions = predict_images_with_cnn(
        model=model,
        image_paths=image_paths,
        transform=transform,
        threshold=threshold,
    )

    for row in predictions:
        print("-" * 70)
        print(f"Input image   : {row['path']}")
        print(f"Predicted     : {row['predicted_class']}")
        print(f"Confidence    : {row['confidence']:.4f}")
        print(f"Threshold     : {row['threshold']:.2f}")
        print(f"Decision      : {row['decision']}")

    save_predictions_table(predictions, run_dir / "predictions.csv")
    save_text(
        "\n".join([
            "Prediction run complete.",
            f"Model: {model_path}",
            f"Inputs evaluated: {len(predictions)}",
            f"Threshold: {threshold:.2f}",
        ]),
        run_dir / "summary.txt",
    )

    print(f"Saved prediction artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
