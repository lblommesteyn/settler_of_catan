"""
ML-based opening quality models.

Logistic regression (interpretable, fast) and gradient boosting (accurate).
Both implement the OpeningModel interface for transparent comparison.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .base_model import OpeningModel
from ..features.opening_features import (
    OpeningFeatures,
    opening_features_to_array,
    FEATURE_NAMES,
)


class LogisticOpeningModel(OpeningModel):
    """
    Logistic regression with feature scaling.
    Fast, interpretable via coefficients, good baseline for ML.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._pipeline = None

    def _make_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter, solver="lbfgs", class_weight="balanced"
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._pipeline = self._make_pipeline()
        self._pipeline.fit(X, y)

    def predict_win_probability(self, features: OpeningFeatures) -> float:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        arr = opening_features_to_array(features).reshape(1, -1)
        return float(self._pipeline.predict_proba(arr)[0, 1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._pipeline.predict_proba(X)[:, 1]

    def feature_importances(self) -> Optional[dict[str, float]]:
        if self._pipeline is None:
            return None
        coefs = self._pipeline.named_steps["clf"].coef_[0]
        return dict(zip(FEATURE_NAMES, coefs.tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipeline, f)

    @classmethod
    def load(cls, path: Path) -> "LogisticOpeningModel":
        model = cls()
        with open(path, "rb") as f:
            model._pipeline = pickle.load(f)
        return model


class GradientBoostingOpeningModel(OpeningModel):
    """
    Gradient boosting model.
    More accurate than logistic regression; feature importances via tree structure.
    Uses sklearn GBC; swap for lightgbm for faster training on large datasets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self._model = None

    def _make_model(self):
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = self._make_model()
        self._model.fit(X, y)

    def predict_win_probability(self, features: OpeningFeatures) -> float:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        arr = opening_features_to_array(features).reshape(1, -1)
        return float(self._model.predict_proba(arr)[0, 1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)[:, 1]

    def feature_importances(self) -> Optional[dict[str, float]]:
        if self._model is None:
            return None
        imps = self._model.feature_importances_
        return dict(zip(FEATURE_NAMES, imps.tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, path: Path) -> "GradientBoostingOpeningModel":
        m = cls()
        with open(path, "rb") as f:
            m._model = pickle.load(f)
        return m


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logreg",
    cv_folds: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Train a model and return evaluation metrics.
    Returns: {model, auc_roc, accuracy, log_loss, feature_importances}
    """
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    if model_type == "logreg":
        model = LogisticOpeningModel()
    elif model_type in ("gbc", "gbm", "lgbm"):
        model = GradientBoostingOpeningModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    probs = model.predict_batch(X_test)

    metrics = {
        "model": model,
        "auc_roc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, (probs > 0.5).astype(int)),
        "log_loss": log_loss(y_test, probs),
        "feature_importances": model.feature_importances(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Cross-validated AUC
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression as LR
        base_clf = LR(C=1.0, max_iter=1000)
        cv_scores = cross_val_score(
            Pipeline([("sc", StandardScaler()), ("clf", base_clf)]),
            X, y, cv=cv_folds, scoring="roc_auc",
        )
        metrics["cv_auc_mean"] = float(cv_scores.mean())
        metrics["cv_auc_std"] = float(cv_scores.std())
    except Exception:
        pass

    return metrics


def compare_all_models(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Train and compare all models. Returns {model_name: metrics}."""
    from .heuristic import PipCountHeuristic, WeightedHeuristic
    from sklearn.metrics import roc_auc_score

    results = {}

    for name in ["logreg", "gbc"]:
        results[name] = train_and_evaluate(X, y, model_type=name, seed=seed)

    # Heuristics (no training)
    for name, heuristic in [("pip_count", PipCountHeuristic()), ("weighted", WeightedHeuristic())]:
        probs = heuristic.predict_batch(X)
        results[name] = {
            "model": heuristic,
            "auc_roc": roc_auc_score(y, probs),
            "n_train": len(X),
        }

    return results
