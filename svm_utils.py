from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET_LABELS = {
    "Linear separation": "Mostly linearly separable clusters",
    "Two moons": "Curved, interlocking classes",
    "Concentric circles": "Nested nonlinear classes",
    "XOR": "Diagonal class pattern that defeats a single straight boundary",
}


@dataclass(frozen=True)
class ModelResults:
    pipeline: Pipeline
    metrics: Dict[str, float]
    confusion: np.ndarray
    support_indices_train: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def make_dataset(name: str, n_samples: int, noise: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a reproducible two-feature binary classification dataset."""
    if name == "Linear separation":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[(-1.6, -1.1), (1.5, 1.1)],
            cluster_std=0.65 + 1.25 * noise,
            random_state=random_state,
        )
    elif name == "Two moons":
        X, y = make_moons(
            n_samples=n_samples,
            noise=max(0.01, noise),
            random_state=random_state,
        )
        X = X * np.array([2.0, 1.7])
    elif name == "Concentric circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=max(0.01, noise),
            factor=0.42,
            random_state=random_state,
        )
        X = X * 2.25
    elif name == "XOR":
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-2.2, 2.2, size=(n_samples, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X = X + rng.normal(0, noise * 0.75, size=X.shape)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def build_svc(kernel: str, C: float, gamma: float, degree: int, coef0: float) -> Pipeline:
    """Create a standardized SVC pipeline."""
    kwargs = {
        "kernel": kernel,
        "C": float(C),
        "probability": True,
        "random_state": 0,
    }
    if kernel in {"rbf", "poly"}:
        kwargs["gamma"] = float(gamma)
    if kernel == "poly":
        kwargs["degree"] = int(degree)
        kwargs["coef0"] = float(coef0)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**kwargs)),
    ])


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    gamma: float,
    degree: int,
    coef0: float,
    test_size: float,
    random_state: int,
) -> ModelResults:
    """Split data, train SVC, and calculate classroom-friendly diagnostics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    model = build_svc(kernel, C, gamma, degree, coef0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    svc = model.named_steps["svc"]
    support_indices_train = svc.support_.copy()
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Support vectors": float(len(support_indices_train)),
        "Support vector share": len(support_indices_train) / len(X_train),
    }
    return ModelResults(
        pipeline=model,
        metrics=metrics,
        confusion=confusion_matrix(y_test, pred, labels=[0, 1]),
        support_indices_train=support_indices_train,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def decision_grid(X: np.ndarray, resolution: int = 240, padding: float = 0.8):
    """Build a bounded 2D mesh for decision-boundary plotting."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(resolution)),
        np.linspace(y_min, y_max, int(resolution)),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def kernel_similarity(distance: np.ndarray, gamma: float) -> np.ndarray:
    """RBF similarity as a function of Euclidean distance."""
    distance = np.asarray(distance, dtype=float)
    return np.exp(-float(gamma) * distance**2)


def poly_response(dot_product: np.ndarray, gamma: float, degree: int, coef0: float) -> np.ndarray:
    """Polynomial kernel response for a range of dot products."""
    dot_product = np.asarray(dot_product, dtype=float)
    return (float(gamma) * dot_product + float(coef0)) ** int(degree)


def metrics_frame(metrics: Dict[str, float]) -> pd.DataFrame:
    """Return core predictive metrics as a tidy dataframe."""
    return pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Value": [metrics[k] for k in ["Accuracy", "Precision", "Recall", "F1"]],
        }
    )
