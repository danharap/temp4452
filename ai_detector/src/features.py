from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.robustness import apply_degradation_pil


def extract_handcrafted_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    gray_hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)
    gray_hist = gray_hist.astype(np.float32)

    channel_means = rgb.reshape(-1, 3).mean(axis=0)
    channel_stds = rgb.reshape(-1, 3).std(axis=0)

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_stats = np.array(
        [
            grad_mag.mean(),
            grad_mag.std(),
            np.percentile(grad_mag, 90),
            np.percentile(grad_mag, 99),
        ],
        dtype=np.float32,
    )

    fft = np.fft.fftshift(np.fft.fft2(gray))
    fft_mag = np.log1p(np.abs(fft))
    h, w = fft_mag.shape
    cy, cx = h // 2, w // 2
    r = max(4, min(h, w) // 8)
    low_region = fft_mag[cy - r : cy + r, cx - r : cx + r]
    low_energy = float(np.mean(low_region))
    total_energy = float(np.mean(fft_mag)) + 1e-8
    high_energy = max(total_energy - low_energy, 0.0)
    freq_stats = np.array([low_energy, high_energy, high_energy / total_energy], dtype=np.float32)

    eps = 1e-8
    entropy = -np.sum(gray_hist * np.log(gray_hist + eps))
    entropy_feature = np.array([entropy], dtype=np.float32)

    feat = np.concatenate(
        [
            gray_hist,
            channel_means.astype(np.float32),
            channel_stds.astype(np.float32),
            grad_stats,
            freq_stats,
            entropy_feature,
        ]
    )
    return feat.astype(np.float32)


def extract_features_from_dataframe(
    dataframe: pd.DataFrame,
    image_size: int = 224,
    degradation: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    paths = []

    for _, row in dataframe.iterrows():
        path = str(row["path"])
        image = Image.open(path).convert("RGB")
        image = image.resize((image_size, image_size))
        image = apply_degradation_pil(image, degradation)

        feat = extract_handcrafted_features(image)
        features.append(feat)
        labels.append(int(row["label"]))
        paths.append(path)

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, paths
