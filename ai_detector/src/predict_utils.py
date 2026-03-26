from pathlib import Path

import torch
from PIL import Image

from src.data_utils import IMAGE_EXTENSIONS


def collect_input_images(input_path: str | Path) -> list[str]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {input_path.suffix}")
        return [str(input_path.resolve())]

    image_paths = []
    for p in input_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(p.resolve()))

    if not image_paths:
        raise RuntimeError(f"No image files found inside: {input_path}")

    return sorted(image_paths)


def predict_images_with_cnn(model, image_paths: list[str], transform, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            logit = model(x).squeeze().item()
            prob_ai = float(torch.sigmoid(torch.tensor(logit)).item())
            pred = 1 if prob_ai >= threshold else 0

            rows.append(
                {
                    "path": path,
                    "prob_ai": prob_ai,
                    "predicted_label": pred,
                    "predicted_class": "ai" if pred == 1 else "real",
                    "confidence": prob_ai if pred == 1 else 1.0 - prob_ai,
                    "threshold": threshold,
                    "decision": "Likely AI-generated" if pred == 1 else "Likely real",
                }
            )

    return rows
