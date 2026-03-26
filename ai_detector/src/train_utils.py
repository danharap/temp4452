import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval_utils import compute_classification_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / max(len(loader.dataset), 1)


def evaluate_model_logits(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_prob, paths = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, labels, batch_paths = batch
                paths.extend(batch_paths)
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, y_true, y_prob, paths


def train_cnn(model, train_loader, val_loader, criterion, optimizer, device, epochs: int, threshold: float = 0.5):
    best_f1 = -1.0
    best_state = None
    history = []
    best_val_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, y_true, y_prob, _ = evaluate_model_logits(model, val_loader, criterion, device)
        val_metrics, _ = compute_classification_metrics(y_true, y_prob, threshold=threshold)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["f1"]),
            "val_roc_auc": val_metrics.get("roc_auc"),
        }
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_metrics


def save_history_csv(history: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_path, index=False)
