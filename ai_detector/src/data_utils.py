from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_MAP = {"real": 0, "ai": 1}


def load_yaml_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_genimage_manifest(
    raw_root: str | Path,
    selected_generators: Optional[list[str]] = None,
    ai_folder_name: str = "ai",
    real_folder_name: str = "nature",
) -> pd.DataFrame:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root does not exist: {raw_root}")

    selected_set = set(selected_generators) if selected_generators else None
    ai_folder_name = ai_folder_name.lower()
    real_folder_name = real_folder_name.lower()

    rows = []
    for image_path in raw_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent_name = image_path.parent.name.lower()
        if parent_name == ai_folder_name:
            label_name = "ai"
        elif parent_name == real_folder_name:
            label_name = "real"
        else:
            continue

        relative_parts = image_path.relative_to(raw_root).parts
        generator = relative_parts[0] if len(relative_parts) > 0 else "unknown"
        if selected_set is not None and generator not in selected_set:
            continue

        split_hint = "unknown"
        for part in relative_parts:
            part_l = part.lower()
            if part_l in {"train", "val", "valid", "validation", "test"}:
                split_hint = "val" if part_l in {"val", "valid", "validation"} else part_l
                break

        rows.append(
            {
                "path": str(image_path.resolve()),
                "label": LABEL_MAP[label_name],
                "label_name": label_name,
                "generator": generator,
                "split_hint": split_hint,
            }
        )

    if not rows:
        raise RuntimeError(
            "No images discovered. Check raw_data_root and ai/nature folder names in config."
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df


def apply_balanced_subset(df: pd.DataFrame, max_per_class: int, seed: int = 42) -> pd.DataFrame:
    class_groups = []
    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        n = min(len(class_df), max_per_class)
        class_groups.append(class_df.sample(n=n, random_state=seed, replace=False))

    balanced = pd.concat(class_groups, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_manifest(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")

    stratify = df["label"] if len(df["label"].unique()) > 1 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=stratify,
    )

    temp_ratio = val_ratio + test_ratio
    val_size_within_temp = val_ratio / temp_ratio if temp_ratio > 0 else 0.5
    stratify_temp = temp_df["label"] if len(temp_df["label"].unique()) > 1 else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_within_temp,
        random_state=seed,
        stratify=stratify_temp,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_dir: str | Path,
) -> None:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)


def load_split_dataframes(split_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "Expected split files train.csv/val.csv/test.csv in "
            f"{split_dir}. Generate them with build_splits_from_manifests.py "
            "or prepare_data.py."
        )

    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)


def summarize_split_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    def summarize(df: pd.DataFrame) -> dict:
        counts = df["label_name"].value_counts().to_dict()
        return {
            "n_samples": int(len(df)),
            "n_real": int(counts.get("real", 0)),
            "n_ai": int(counts.get("ai", 0)),
        }

    return {
        "train": summarize(train_df),
        "val": summarize(val_df),
        "test": summarize(test_df),
        "total": int(len(train_df) + len(val_df) + len(test_df)),
    }
