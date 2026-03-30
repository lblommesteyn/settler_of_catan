"""Abstract base class for opening quality models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..features.opening_features import OpeningFeatures, opening_features_to_array


class OpeningModel(ABC):
    """Common interface for all opening evaluators."""

    @abstractmethod
    def predict_win_probability(self, features: OpeningFeatures) -> float:
        """Estimate win probability for one opening. Returns float in [0, 1]."""

    @abstractmethod
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict win probability for a batch of feature arrays.
        X shape: (N, D). Returns shape (N,).
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on feature matrix X (N, D) and binary labels y (N,)."""

    def rank_openings(
        self,
        candidates: list[OpeningFeatures],
    ) -> list[tuple[OpeningFeatures, float]]:
        """Return candidates sorted by predicted win probability (descending)."""
        if not candidates:
            return []
        X = np.stack([opening_features_to_array(f) for f in candidates])
        scores = self.predict_batch(X)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked

    def feature_importances(self) -> Optional[dict[str, float]]:
        """Return {feature_name: importance}. None if not supported."""
        return None
