import argparse
from pathlib import Path

from src.data_utils import (
    apply_balanced_subset,
    discover_genimage_manifest,
    load_yaml_config,
    save_split_dataframes,
    split_manifest,
    summarize_split_counts,
)
from src.run_utils import create_run_dir, save_json, save_text, save_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GenImage subset and train/val/test splits.")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    run_dir = create_run_dir("prepare_data")
    save_yaml(cfg, run_dir / "config_used.yaml")

    manifest = discover_genimage_manifest(
        raw_root=cfg["raw_data_root"], # UPDATE THIS
        selected_generators=cfg.get("selected_generators"),
        ai_folder_name=cfg.get("ai_folder_name", "ai"),
        real_folder_name=cfg.get("real_folder_name", "nature"),
    )

    data_dir = Path(cfg["output"].get("data_dir", "data"))
    split_dir = Path(cfg["output"].get("split_dir", "data/splits"))
    data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(data_dir / cfg["output"].get("manifest_filename", "manifest_full.csv"), index=False)

    subset_cfg = cfg.get("subset", {})
    if subset_cfg.get("enabled", True):
        manifest = apply_balanced_subset(
            manifest,
            max_per_class=int(subset_cfg.get("max_per_class", 4000)),
            seed=int(cfg.get("seed", 42)),
        )

    manifest.to_csv(data_dir / cfg["output"].get("subset_manifest_filename", "manifest_subset.csv"), index=False)

    split_cfg = cfg["split"]
    train_df, val_df, test_df = split_manifest(
        manifest,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(cfg.get("seed", 42)),
    )

    save_split_dataframes(train_df, val_df, test_df, split_dir)
    split_summary = summarize_split_counts(train_df, val_df, test_df)

    save_json(split_summary, run_dir / "metrics.json")
    save_text(
        "\n".join([
            "Dataset preparation complete.",
            f"Total selected images: {len(manifest)}",
            f"Train: {len(train_df)}",
            f"Val: {len(val_df)}",
            f"Test: {len(test_df)}",
        ]),
        run_dir / "summary.txt",
    )

    print("Preparation complete.")
    print(f"Saved split files to: {split_dir}")
    print(split_summary)


if __name__ == "__main__":
    main()
