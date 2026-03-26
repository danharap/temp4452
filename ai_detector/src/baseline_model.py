from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 3000,
    seed: int = 42,
):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    return np.asarray(probs, dtype=float)


def save_baseline_model(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_baseline_model(model_path: str | Path):
    return joblib.load(model_path)
