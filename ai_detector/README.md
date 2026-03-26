# AI-Image Detector (Course Project)

Terminal-based deep learning computer vision project for binary classification:

- `real` (natural images)
- `ai` (AI-generated images)

## Quick Setup (For TA Reproducibility)

Use these commands from project root to reproduce the main pipeline quickly.

1. Download the dataset zip from the provided Google Drive link: DOUBLE CHECK THIS!!!!!!!!!!!!!!!!!!!!!! and put it in teh detailed setup too:https://drive.google.com/file/d/1vRsyp7OmTveFyT14HiiMLFbZ5NSKJO5I/view?usp=share_link

2. Place the zip inside `data/` and extract it there.
   After extraction, `data/` should contain:

```text
data/
  genimage_splits/
  splits/
  test_images/
```

3. Set up environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

4. Build split CSVs from finalized manifests (no reshuffle):

```bash
python3 build_splits_from_manifests.py --source-root data/genimage_splits --output-dir data/splits
```

5. Train and evaluate classical baseline:

```bash
python3 train_baseline.py --config configs/baseline_config.yaml
```

6. Train and evaluate CNN:

```bash
python3 train_cnn.py --config configs/cnn_config.yaml
```

7. Evaluate robustness on clean/JPEG/resize variants:

```bash
python3 evaluate.py --config configs/eval_config.yaml
```

8. Run final detector inference on new input image(s):

```bash
python3 predict.py --config configs/predict_config.yaml --input /path/to/image_or_folder
```

Notes:

- Run steps 4-6 before step 7 or 8 if no trained model exists yet.
- `evaluate.py` and `predict.py` assume a trained model already exists at the configured model path.
- Re-running commands creates new timestamped folders under `runs/`; previous runs do not need to be deleted.
- `test_images/` is only for a small prediction smoke test, not for training.
- Tested with Python 3.14.3.

## What This Project Uses

- **Dataset pipeline**
  - Handles raw GenImage-style discovery and optional subset/splitting.
  - Also supports pre-split datasets by building `data/splits/*.csv` from provided manifests.
- **Classical baseline**
  - Handcrafted image features + logistic regression.
  - Useful as a comparison anchor for the CNN.
- **Deep model**
  - Transfer-learned ResNet binary classifier.
  - Trained on `train.csv`, validated on `val.csv`, tested on `test.csv`.
- **Robustness evaluation**
  - Evaluates clean and degraded test variants (JPEG compression, resizing).
- **Prediction flow**
  - Runs inference on one image or a folder, outputs class + confidence.
- **Artifact-first runs**
  - Every major command creates a run folder under `runs/` with configs, metrics, and outputs.

## Project Tree and File Purpose

```text
ai_detector/
├── build_splits_from_manifests.py   # Build data/splits/*.csv from existing split_manifest.csv files
├── prepare_data.py                  # Build manifest/subset/splits from raw GenImage-style folders
├── train_baseline.py                # Train + evaluate classical baseline
├── train_cnn.py                     # Train + evaluate transfer-learned CNN
├── evaluate.py                      # Evaluate a trained model on clean + degraded test data
├── predict.py                       # Run inference on user input image(s)
│
├── configs/
│   ├── data_config.yaml             # Dataset discovery/subset/split settings for prepare_data.py
│   ├── baseline_config.yaml         # Baseline training settings
│   ├── cnn_config.yaml              # CNN training settings (epochs, batch size, workers, etc.)
│   ├── eval_config.yaml             # Evaluation settings (model type/path, degradations)
│   └── predict_config.yaml          # Prediction defaults (model path, threshold)
│
├── src/
│   ├── data_utils.py                # Data discovery, split save/load, config loading
│   ├── dataset.py                   # PyTorch dataset + transforms
│   ├── features.py                  # Handcrafted feature extraction
│   ├── baseline_model.py            # Baseline model train/save/load/predict
│   ├── cnn_model.py                 # ResNet binary classifier builder
│   ├── train_utils.py               # Training/validation loops and history
│   ├── eval_utils.py                # Metrics, plots, prediction export, error examples
│   ├── robustness.py                # JPEG/resize degradations
│   ├── predict_utils.py             # Prediction input handling and inference helpers
│   └── run_utils.py                 # Run directory and artifact saving helpers
│
├── data/                            # Split CSVs and/or dataset folders (user provided)
├── models/                          # Saved main model artifacts (baseline/cnn)
└── runs/                            # Timestamped run artifacts for train/eval/predict
```

## Detailed Setup

### 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2) Data download and placement (TA flow)

- The repository will be provided with an empty `data/` folder.
- Download the dataset zip from the provided Google Drive link.
- Move the zip into `data/` and extract it there.

Expected structure after extraction:

```text
data/
  genimage_splits/
  splits/
  test_images/
```

`test_images/` is intended for quick prediction smoke tests only (with `predict.py`) and is not required for training.

### 3) Current dataset setup and counts

This repository is currently set up to use split folders under `data/genimage_splits/`:

- `biggan_16k_split`
- `midjourney_16k_split`

Each generator split follows:

```text
data/genimage_splits/
  your_generator_16k_split/
    split_manifest.csv
    train/
      ai/
      nature/
    val/
      ai/
      nature/
    test/
      ai/
      nature/
```

Project split CSVs are generated into:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Current combined split counts (both generators together):

- `train.csv`: 22,400 images (11,200 per generator)
- `val.csv`: 4,800 images (2,400 per generator)
- `test.csv`: 4,800 images (2,400 per generator)

## How To Add Your Own Images

Choose one path below depending on your dataset state.

### Option A: You already have fixed split folders (recommended if splits are finalized)

Put your data under `data/genimage_splits/<generator_name>_split/` with:

```text
data/genimage_splits/
  your_generator_16k_split/
    split_manifest.csv
    train/
      ai/
      nature/
    val/
      ai/
      nature/
    test/
      ai/
      nature/
```

Then generate project split CSVs (without reshuffling):

```bash
python3 build_splits_from_manifests.py --source-root data/genimage_splits --output-dir data/splits
```

### Option B: You have raw folders and want this repo to create splits

Use raw structure where class folders are discoverable (`ai` and `nature`, by default), then:

```bash
python3 prepare_data.py --config configs/data_config.yaml
```

Use this only if you want the project to create subset/splits. If your splits are already finalized, use Option A.

## Main Workflow (What Each Command Does)

### 1) Build split CSVs

If using finalized split manifests:

```bash
python3 build_splits_from_manifests.py --source-root data/genimage_splits --output-dir data/splits
```

Creates:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Each row includes image `path` and class `label`.

### 2) Train classical baseline

```bash
python3 train_baseline.py --config configs/baseline_config.yaml
```

Does:

- Extract handcrafted features
- Train logistic regression baseline
- Evaluate on val/test
- Save model + metrics + plots to `runs/baseline_train_<timestamp>/`

### 3) Train CNN

```bash
python3 train_cnn.py --config configs/cnn_config.yaml
```

Does:

- Build transfer-learned ResNet binary classifier
- Train on train split, track val metrics each epoch
- Select best model by validation F1
- Evaluate on test
- Save `models/cnn_model.pt` and full run artifacts

### 4) Evaluate robustness

```bash
python3 evaluate.py --config configs/eval_config.yaml
```

Does:

- Evaluate trained model on clean test data
- Evaluate on configured degradations (JPEG/resize)
- Save metrics, plots, and misclassification examples

### 5) Predict on new input

```bash
python3 predict.py --config configs/predict_config.yaml --input /path/to/image_or_folder
```

Does:

- Load trained CNN
- Predict class + confidence for each input image
- Save prediction run artifacts

## Where Outputs Are Saved

- Split metadata and split CSVs: `data/`
- Main model files: `models/`
- Run artifacts: `runs/<run_type>_<timestamp>/`

Typical run folder artifacts:

- `config_used.yaml`
- `metrics.json`
- `summary.txt`
- plots (confusion matrix, ROC)
- per-image predictions
- training history/model (for training runs)
