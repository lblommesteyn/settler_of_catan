"""
Heuristic baseline models.

These require no training data and encode domain knowledge directly.
They serve as interpretable baselines to compare against ML models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_model import OpeningModel
from ..features.opening_features import (
    OpeningFeatures,
    opening_features_to_array,
    FEATURE_NAMES,
    N_OPENING_FEATURES,
)


class PipCountHeuristic(OpeningModel):
    """
    Simplest baseline: score = combined pip count / max possible pips.
    Directly tests Hypothesis H1: raw pip count is a weak predictor.
    """
    # Theoretical max: two spots with 5+5+5 pips (two 6/8/5 triple hexes)
    MAX_PIPS = 30.0

    def predict_win_probability(self, features: OpeningFeatures) -> float:
        return min(features.combined_pip_count / self.MAX_PIPS, 1.0)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        pip_idx = FEATURE_NAMES.index("combined_pip_count")
        return np.clip(X[:, pip_idx] / self.MAX_PIPS, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass  # no training needed


class WeightedHeuristic(OpeningModel):
    """
    Weighted linear combination of interpretable features, normalized via sigmoid.
    Weights encode known Catan domain knowledge.

    This is a strong domain-expert baseline: outperforming it with ML is meaningful.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "combined_pip_count": 0.35,
        "unique_resource_count": 0.20,
        "expansion_vertex_count": 0.15,
        "expansion_pip_sum": 0.10,
        "port_synergy_score": 0.10,
        "has_all_five": 0.05,
        "combined_robber_vulnerability": -0.05,  # negative: high exposure is bad
    }

    # Normalization denominators (approximate max value for each feature)
    FEATURE_NORMS: dict[str, float] = {
        "combined_pip_count": 30.0,
        "unique_resource_count": 5.0,
        "expansion_vertex_count": 15.0,
        "expansion_pip_sum": 60.0,
        "port_synergy_score": 8.0,
        "has_all_five": 1.0,
        "combined_robber_vulnerability": 1.0,
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def _score(self, features: OpeningFeatures) -> float:
        feature_vals = {
            "combined_pip_count": features.combined_pip_count,
            "unique_resource_count": features.unique_resource_count,
            "expansion_vertex_count": features.expansion_vertex_count,
            "expansion_pip_sum": features.expansion_pip_sum,
            "port_synergy_score": features.port_synergy_score,
            "has_all_five": float(features.has_all_five),
            "combined_robber_vulnerability": features.combined_robber_vulnerability,
        }
        score = 0.0
        for feat, w in self.weights.items():
            val = feature_vals.get(feat, 0.0)
            norm = self.FEATURE_NORMS.get(feat, 1.0)
            score += w * (val / max(norm, 1e-9))
        return score

    def _sigmoid(self, x: float) -> float:
        import math
        # Shift so score=0 → prob=0.25 (random for 4 players)
        return 1.0 / (1.0 + math.exp(-4.0 * (x - 0.5)))

    def predict_win_probability(self, features: OpeningFeatures) -> float:
        return self._sigmoid(self._score(features))

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        # Map feature names to indices
        name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        scores = np.zeros(len(X))
        for feat, w in self.weights.items():
            if feat in name_to_idx:
                idx = name_to_idx[feat]
                norm = self.FEATURE_NORMS.get(feat, 1.0)
                scores += w * (X[:, idx] / max(norm, 1e-9))
        # Sigmoid
        return 1.0 / (1.0 + np.exp(-4.0 * (scores - 0.5)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass  # hand-tuned; no data-driven fitting in base version

    def feature_importances(self) -> dict[str, float]:
        return {k: abs(v) for k, v in self.weights.items()}
