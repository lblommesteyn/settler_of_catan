"""
Opening explanation generator.

Turns a model score + feature vector into a human-readable explanation:
- Win probability and percentile
- Archetype classification
- Top strength/weakness reasons (template-based)
- Counterfactual alternatives
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..board.board import CatanBoard
from ..features.opening_features import (
    OpeningFeatures,
    opening_features_to_array,
    identify_archetype,
    FEATURE_NAMES,
    compute_opening_features,
    compute_all_vertex_features,
)
from ..models.base_model import OpeningModel


ARCHETYPE_DESCRIPTIONS = {
    "ore_wheat": "Ore/Wheat Engine",
    "road_race": "Road Race",
    "port_engine": "Port Engine",
    "balanced": "Balanced Expansion",
    "high_pip": "High Production",
}


@dataclass
class Counterfactual:
    v1: tuple[float, float]
    v2: tuple[float, float]
    win_prob: float
    delta: float          # win_prob - current opening's win_prob
    gain_desc: str        # e.g. "Better ore/wheat synergy"
    cost_desc: str        # e.g. "Lower raw pip count"


@dataclass
class OpeningExplanation:
    win_probability: float
    percentile: float         # 0-100
    archetype: str            # e.g. "Balanced Expansion"
    strengths: list[str]      # up to 3 positive bullets
    weaknesses: list[str]     # up to 3 negative bullets
    feature_contributions: dict[str, float]
    counterfactuals: list[Counterfactual]  # top alternatives
    raw_features: OpeningFeatures


def explain_opening(
    features: OpeningFeatures,
    model: OpeningModel,
    board: CatanBoard,
    top_k_counterfactuals: int = 3,
    sample_alternatives: int = 30,
) -> OpeningExplanation:
    """Generate full explanation for one opening."""
    win_prob = model.predict_win_probability(features)

    # --- Percentile against random sample of alternatives ---
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    percentile = _compute_percentile(features, win_prob, model, board, vf_cache, legal)

    # --- Archetype ---
    archetype_key = identify_archetype(features)
    archetype = ARCHETYPE_DESCRIPTIONS.get(archetype_key, archetype_key)

    # --- Feature contributions ---
    contributions = _compute_contributions(features, model)

    # --- Strengths and weaknesses ---
    strengths = _generate_strengths(features, contributions)
    weaknesses = _generate_weaknesses(features, contributions)

    # --- Counterfactuals ---
    cfs = _find_counterfactuals(
        features, win_prob, model, board, vf_cache, legal,
        top_k=top_k_counterfactuals,
        n_sample=sample_alternatives,
    )

    return OpeningExplanation(
        win_probability=win_prob,
        percentile=percentile,
        archetype=archetype,
        strengths=strengths,
        weaknesses=weaknesses,
        feature_contributions=contributions,
        counterfactuals=cfs,
        raw_features=features,
    )


def _compute_percentile(
    features: OpeningFeatures,
    win_prob: float,
    model: OpeningModel,
    board: CatanBoard,
    vf_cache: dict,
    legal: list,
    n_sample: int = 50,
) -> float:
    """Estimate percentile vs. random sample of other legal openings."""
    import random
    rng = random.Random(42)

    other_probs = []
    candidates = legal[:]
    for _ in range(min(n_sample, len(candidates))):
        v1 = rng.choice(candidates)
        v2_cands = board.legal_second_vertices(v1)
        if not v2_cands:
            continue
        v2 = rng.choice(v2_cands)
        try:
            other_f = compute_opening_features(v1, v2, features.seat, board, vf_cache)
            other_probs.append(model.predict_win_probability(other_f))
        except Exception:
            pass

    if not other_probs:
        return 50.0
    below = sum(1 for p in other_probs if p <= win_prob)
    return round(100.0 * below / len(other_probs), 1)


def _compute_contributions(
    features: OpeningFeatures,
    model: OpeningModel,
) -> dict[str, float]:
    """
    Estimate feature contributions.
    For logistic: coef * standardized value (approximate).
    For others: return raw normalized feature values weighted by importance.
    """
    arr = opening_features_to_array(features)
    importances = model.feature_importances()

    if importances is None:
        return {}

    # For gradient boosting, importances are magnitudes; just return them
    return {k: float(v) for k, v in importances.items()}


def _generate_strengths(
    features: OpeningFeatures,
    contributions: dict[str, float],
) -> list[str]:
    """Generate up to 3 human-readable strength bullets."""
    reasons = []

    # Production
    if features.combined_pip_count >= 15:
        reasons.append(
            f"Strong production: {features.combined_pip_count} pips across "
            f"{features.unique_resource_count} resources"
        )
    elif features.combined_pip_count >= 12:
        reasons.append(f"Good production: {features.combined_pip_count} pips")

    # Diversity
    if features.has_all_five:
        reasons.append("Full resource coverage: all 5 resources accessible")
    elif features.unique_resource_count == 4:
        reasons.append(f"Wide resource coverage: 4 distinct resources")

    # Expansion
    if features.expansion_vertex_count >= 8:
        reasons.append(
            f"Excellent expansion optionality: {features.expansion_vertex_count} reachable spots"
        )
    elif features.expansion_vertex_count >= 5:
        reasons.append(f"Good expansion lanes: {features.expansion_vertex_count} accessible vertices")

    # Port synergy
    if features.has_specific_port and features.port_synergy_score >= 3.0:
        reasons.append(
            f"Strong port synergy: {_port_type_str(features)} with matching production"
        )
    elif features.num_ports > 0 and features.best_port_ratio >= 0.4:
        reasons.append("Port access provides trading efficiency")

    # City path
    if features.combined_city_pips >= 8:
        reasons.append(
            f"Strong city development path: {features.combined_ore_pips} ore + "
            f"{features.combined_wheat_pips} wheat pips"
        )

    return reasons[:3]


def _generate_weaknesses(
    features: OpeningFeatures,
    contributions: dict[str, float],
) -> list[str]:
    """Generate up to 3 human-readable weakness bullets."""
    reasons = []

    # Low production
    if features.combined_pip_count < 10:
        reasons.append(
            f"Low production: only {features.combined_pip_count} total pips"
        )

    # Poor diversity
    if features.unique_resource_count <= 2:
        reasons.append(
            f"Narrow resources: only {features.unique_resource_count} distinct type(s)"
        )
    if features.city_resource_count < 2:
        reasons.append("Missing ore or wheat — city development will require trading")

    # Poor expansion
    if features.expansion_vertex_count <= 2:
        reasons.append(
            f"Limited expansion: only {features.expansion_vertex_count} reachable vertices"
        )

    # Robber exposure
    if features.combined_robber_vulnerability >= 0.5:
        reasons.append(
            "High robber exposure: both settlements on frequently-targeted hexes"
        )
    elif features.double_exposure_count >= 1:
        reasons.append(
            f"{features.double_exposure_count} hex(es) adjacent to both settlements (robber double-hit)"
        )

    # No port
    if features.num_ports == 0 and features.combined_pip_count >= 13:
        reasons.append("No port access — trading efficiency will suffer mid-game")

    return reasons[:3]


def _port_type_str(features: OpeningFeatures) -> str:
    """Return a human-readable port description."""
    from ..features.vertex_features import VertexFeatures
    # Check both vertices' port types via the OpeningFeatures arrays
    # (use feature values as proxy — port_type is encoded as port_ratio in array)
    if features.best_port_ratio == 0.5:
        return "2:1 resource port"
    return "3:1 generic port"


def _find_counterfactuals(
    features: OpeningFeatures,
    win_prob: float,
    model: OpeningModel,
    board: CatanBoard,
    vf_cache: dict,
    legal: list,
    top_k: int = 3,
    n_sample: int = 30,
) -> list[Counterfactual]:
    """Find top alternatives with higher predicted win probability."""
    import random
    rng = random.Random(123)

    candidates = []
    for _ in range(n_sample):
        v1 = rng.choice(legal)
        v2_cands = board.legal_second_vertices(v1)
        if not v2_cands:
            continue
        v2 = rng.choice(v2_cands)
        # Skip if same opening
        if {v1, v2} == {features.v1, features.v2}:
            continue
        try:
            alt_f = compute_opening_features(v1, v2, features.seat, board, vf_cache)
            alt_prob = model.predict_win_probability(alt_f)
            delta = alt_prob - win_prob
            if delta > 0:
                gain = _describe_gain(features, alt_f)
                cost = _describe_cost(features, alt_f)
                candidates.append(Counterfactual(
                    v1=v1, v2=v2,
                    win_prob=alt_prob,
                    delta=delta,
                    gain_desc=gain,
                    cost_desc=cost,
                ))
        except Exception:
            pass

    candidates.sort(key=lambda c: c.delta, reverse=True)
    return candidates[:top_k]


def _describe_gain(current: OpeningFeatures, alt: OpeningFeatures) -> str:
    if alt.expansion_vertex_count > current.expansion_vertex_count + 2:
        return f"Better expansion ({alt.expansion_vertex_count} vs {current.expansion_vertex_count} spots)"
    if alt.combined_city_pips > current.combined_city_pips + 2:
        return "Stronger city development path"
    if alt.num_ports > current.num_ports:
        return "Gains port access"
    if alt.unique_resource_count > current.unique_resource_count:
        return f"Better resource diversity ({alt.unique_resource_count} vs {current.unique_resource_count})"
    return "Higher overall opening quality"


def _describe_cost(current: OpeningFeatures, alt: OpeningFeatures) -> str:
    if alt.combined_pip_count < current.combined_pip_count - 2:
        return f"Lower raw production ({alt.combined_pip_count} vs {current.combined_pip_count} pips)"
    if alt.combined_robber_vulnerability > current.combined_robber_vulnerability + 0.1:
        return "Higher robber vulnerability"
    if alt.expansion_vertex_count < current.expansion_vertex_count:
        return "Fewer expansion options"
    return "Minor production trade-off"
