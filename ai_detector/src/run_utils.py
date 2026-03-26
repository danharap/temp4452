import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


RUNS_DIR = Path("runs")


def create_run_dir(run_type: str, base_dir: Path | str = RUNS_DIR) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_path / f"{run_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data: Any, output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_yaml(data: Any, output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_text(text: str, output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def save_predictions_table(rows: Iterable[Mapping[str, Any]], output_path: Path | str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
