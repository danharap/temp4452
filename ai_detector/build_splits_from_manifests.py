import argparse
from pathlib import Path

import pandas as pd


LABEL_MAP = {"ai": 1, "nature": 0}
EXPECTED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/splits CSVs from existing finalized split manifests."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/genimage_splits",
        help="Folder containing per-generator *_split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Folder where train.csv/val.csv/test.csv will be written.",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path, source_root: Path) -> list[dict]:
    generator_root = manifest_path.parent
    df = pd.read_csv(manifest_path)

    required_cols = {"label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {manifest_path}")

    rows: list[dict] = []
    for rec in df.to_dict(orient="records"):
        split = str(rec["split"]).strip().lower()
        label_name = str(rec["label"]).strip().lower()
        source_path = str(rec.get("source_path", "")).strip()
        dest_path = str(rec.get("dest_path", "")).strip()

        if split not in EXPECTED_SPLITS:
            continue
        if label_name not in LABEL_MAP:
            continue

        # Some manifests store the finalized location in dest_path.
        # Prefer that first so we respect existing split folders exactly.
        candidate_paths = []
        if dest_path:
            candidate_paths.append((source_root / dest_path).resolve())
            candidate_paths.append((generator_root / dest_path).resolve())
        if source_path:
            candidate_paths.append((generator_root / source_path).resolve())

        image_path = next((p for p in candidate_paths if p.exists()), None)
        if image_path is None:
            image_path = candidate_paths[0] if candidate_paths else generator_root.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Referenced image does not exist: {image_path}")

        rows.append(
            {
                "path": str(image_path),
                "label": LABEL_MAP[label_name],
                "label_name": label_name,
                "generator": generator_root.name,
                "split": split,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)

    manifest_paths = sorted(source_root.glob("*_split/split_manifest.csv"))
    if not manifest_paths:
        raise FileNotFoundError(f"No split manifests found in {source_root}")

    all_rows: list[dict] = []
    for manifest_path in manifest_paths:
        all_rows.extend(load_manifest_rows(manifest_path, source_root=source_root))

    if not all_rows:
        raise RuntimeError("No valid rows found in manifests.")

    merged = pd.DataFrame(all_rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in EXPECTED_SPLITS:
        split_df = merged[merged["split"] == split].copy()
        split_df = split_df.sort_values("path").reset_index(drop=True)
        split_df.to_csv(output_dir / f"{split}.csv", index=False)

    print("Done.")
    print(f"Wrote: {output_dir / 'train.csv'}")
    print(f"Wrote: {output_dir / 'val.csv'}")
    print(f"Wrote: {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
