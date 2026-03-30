"""
Opening-pair feature computation.

An opening is defined by (v1, v2, seat): the two initial settlement vertices
and the player's seat position (0=first, 3=last in snake draft).

Features capture both individual vertex quality and pair-level interactions:
expansion optionality, port synergy, resource complementarity, and robber exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..board.board import CatanBoard, Resource
from .vertex_features import (
    VertexFeatures,
    compute_all_vertex_features,
    vertex_features_to_array,
    N_VERTEX_FEATURES,
)


ARCHETYPE_NAMES = ["balanced", "ore_wheat", "road_race", "port_engine", "high_pip"]


@dataclass
class OpeningFeatures:
    v1: tuple[float, float]
    v2: tuple[float, float]
    seat: int  # 0–3

    # --- Combined production ---
    combined_pip_count: int
    unique_resource_count: int        # distinct resources across both settlements
    has_all_five: bool
    settlement_resource_count: int    # unique among {wood, brick, sheep, wheat}
    city_resource_count: int          # unique among {ore, wheat}
    combined_wood_pips: int
    combined_brick_pips: int
    combined_sheep_pips: int
    combined_wheat_pips: int
    combined_ore_pips: int
    combined_food_pips: int
    combined_city_pips: int
    combined_settlement_pips: int

    # --- Port features ---
    num_ports: int                    # 0, 1, or 2
    has_specific_port: bool           # at least one 2:1 port
    best_port_ratio: float            # 0, 1/3, or 1/2
    port_synergy_score: float         # pip-weighted port value

    # --- Expansion / connectivity ---
    vertex_road_distance: int         # road steps between v1 and v2
    expansion_vertex_count: int       # legal vertices reachable in 2 road steps
    expansion_pip_sum: int            # total pips at expansion vertices
    expansion_pip_mean: float
    shared_expansion_hexes: int       # hexes reachable from both settlements (contested)

    # --- Robber vulnerability ---
    combined_robber_vulnerability: float
    double_exposure_count: int        # hexes adjacent to BOTH settlements

    # --- Seat / tempo ---
    seat: int
    is_early_seat: bool               # seat 0 or 1 (places first and fourth)
    # Snake draft: seat 0 picks 1st + last, seat 3 picks 4th + 5th (middle timing)

    # --- Archetype scores (interpretable summaries) ---
    ore_wheat_score: float
    road_race_score: float
    port_engine_score: float
    balanced_score: float
    high_pip_score: float

    # --- Raw sub-vectors (for ML) ---
    v1_arr: np.ndarray   # shape (N_VERTEX_FEATURES,)
    v2_arr: np.ndarray   # shape (N_VERTEX_FEATURES,)


def _expansion_vertices(
    v1: tuple[float, float],
    v2: tuple[float, float],
    board: CatanBoard,
    max_steps: int = 2,
) -> list[tuple[float, float]]:
    """
    Vertices reachable in ≤ max_steps road steps from v1 or v2,
    excluding vertices too close to v1 or v2 (distance rule: within 1 step).
    """
    # Vertices that violate the distance rule relative to own settlements
    too_close: set[tuple[float, float]] = {v1, v2}
    for v in (v1, v2):
        too_close.update(board.graph.vertex_neighbors[v])

    reachable = board.graph.reachable_vertices(
        sources=[v1, v2],
        max_steps=max_steps,
        exclude=too_close,
    )
    # Must also have at least one adjacent hex (not isolated/ghost vertex)
    return [v for v in reachable if board.vertex_hexes.get(v)]


def compute_opening_features(
    v1: tuple[float, float],
    v2: tuple[float, float],
    seat: int,
    board: CatanBoard,
    vf_cache: Optional[dict[tuple[float, float], VertexFeatures]] = None,
) -> OpeningFeatures:
    """Compute the full opening feature vector for (v1, v2, seat)."""
    if vf_cache is None:
        vf_cache = compute_all_vertex_features(board)

    f1 = vf_cache[v1]
    f2 = vf_cache[v2]

    # --- Combined production ---
    res_union: set[Resource] = board.resources_at(v1) | board.resources_at(v2)
    unique_resource_count = len(res_union)
    city_res = {Resource.ORE, Resource.WHEAT}
    settlement_res = {Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT}

    combined_wood = f1.wood_pips + f2.wood_pips
    combined_brick = f1.brick_pips + f2.brick_pips
    combined_sheep = f1.sheep_pips + f2.sheep_pips
    combined_wheat = f1.wheat_pips + f2.wheat_pips
    combined_ore = f1.ore_pips + f2.ore_pips
    combined_food = combined_wheat + combined_sheep
    combined_city = combined_ore + combined_wheat
    combined_settlement = combined_wood + combined_brick + combined_sheep + combined_wheat

    # --- Port features ---
    ports = [p for v in (v1, v2) for p in [board.get_port(v)] if p is not None]
    port_ratios = [p.ratio for p in ports]
    best_ratio = max(port_ratios) if port_ratios else 0.0

    # Port synergy: (pip-weighted) how much relevant resource do we have?
    port_synergy = 0.0
    for port in ports:
        if port.resource is None:
            port_synergy += (f1.total_pips + f2.total_pips) * port.ratio
        else:
            res_pips = {
                Resource.WOOD: combined_wood, Resource.BRICK: combined_brick,
                Resource.SHEEP: combined_sheep, Resource.WHEAT: combined_wheat,
                Resource.ORE: combined_ore,
            }.get(port.resource, 0)
            port_synergy += res_pips * port.ratio

    # --- Expansion ---
    exp_verts = _expansion_vertices(v1, v2, board)
    exp_pip_sum = sum(board.pip_count(v) for v in exp_verts)
    exp_pip_mean = exp_pip_sum / max(len(exp_verts), 1)

    # Hexes reachable from both settlements (double-exposed to block/contention)
    hexes1 = {hc for t in board.vertex_hexes.get(v1, []) for hc in [(t.q, t.r)]}
    hexes2 = {hc for t in board.vertex_hexes.get(v2, []) for hc in [(t.q, t.r)]}
    double_exp = len(hexes1 & hexes2)

    road_dist = board.graph.distance_between(v1, v2)

    # Shared expansion hexes (hexes adjacent to expansion vertices, contested)
    exp_hex_counts: dict[tuple[int, int], int] = {}
    for ev in exp_verts:
        for tile in board.vertex_hexes.get(ev, []):
            k = (tile.q, tile.r)
            exp_hex_counts[k] = exp_hex_counts.get(k, 0) + 1
    shared_exp_hexes = sum(1 for c in exp_hex_counts.values() if c > 1)

    # --- Seat features ---
    is_early = seat in (0, 1)

    # --- Archetype scores ---
    total_pips = f1.total_pips + f2.total_pips
    ore_wheat = combined_city / max(total_pips, 1)
    road_race = combined_settlement / max(total_pips, 1) * (len(exp_verts) / 10)
    port_eng = port_synergy / max(total_pips, 1)
    balanced = _resource_entropy({
        Resource.WOOD: combined_wood, Resource.BRICK: combined_brick,
        Resource.SHEEP: combined_sheep, Resource.WHEAT: combined_wheat,
        Resource.ORE: combined_ore,
    })
    high_pip_score = total_pips / 30.0  # 30 is near-max for two settlements

    return OpeningFeatures(
        v1=v1, v2=v2, seat=seat,
        combined_pip_count=total_pips,
        unique_resource_count=unique_resource_count,
        has_all_five=unique_resource_count == 5,
        settlement_resource_count=len(res_union & settlement_res),
        city_resource_count=len(res_union & city_res),
        combined_wood_pips=combined_wood,
        combined_brick_pips=combined_brick,
        combined_sheep_pips=combined_sheep,
        combined_wheat_pips=combined_wheat,
        combined_ore_pips=combined_ore,
        combined_food_pips=combined_food,
        combined_city_pips=combined_city,
        combined_settlement_pips=combined_settlement,
        num_ports=len(ports),
        has_specific_port=any(p.resource is not None for p in ports),
        best_port_ratio=best_ratio,
        port_synergy_score=port_synergy,
        vertex_road_distance=road_dist,
        expansion_vertex_count=len(exp_verts),
        expansion_pip_sum=exp_pip_sum,
        expansion_pip_mean=exp_pip_mean,
        shared_expansion_hexes=shared_exp_hexes,
        combined_robber_vulnerability=f1.robber_vulnerability + f2.robber_vulnerability,
        double_exposure_count=double_exp,
        is_early_seat=is_early,
        ore_wheat_score=ore_wheat,
        road_race_score=road_race,
        port_engine_score=port_eng,
        balanced_score=balanced,
        high_pip_score=high_pip_score,
        v1_arr=vertex_features_to_array(f1),
        v2_arr=vertex_features_to_array(f2),
    )


def _resource_entropy(pip_by_res: dict[Resource, int]) -> float:
    total = sum(pip_by_res.values())
    if total == 0:
        return 0.0
    probs = [p / total for p in pip_by_res.values() if p > 0]
    return -sum(p * math.log2(p) for p in probs) / math.log2(max(len(probs), 2))


# ---- Feature array for ML ----
# Pair-level scalar features (order must stay stable)
_PAIR_FEATURE_NAMES = [
    "combined_pip_count", "unique_resource_count", "has_all_five",
    "settlement_resource_count", "city_resource_count",
    "combined_wood_pips", "combined_brick_pips", "combined_sheep_pips",
    "combined_wheat_pips", "combined_ore_pips",
    "combined_food_pips", "combined_city_pips", "combined_settlement_pips",
    "num_ports", "has_specific_port", "best_port_ratio", "port_synergy_score",
    "vertex_road_distance", "expansion_vertex_count",
    "expansion_pip_sum", "expansion_pip_mean", "shared_expansion_hexes",
    "combined_robber_vulnerability", "double_exposure_count",
    "seat", "is_early_seat",
    "ore_wheat_score", "road_race_score", "port_engine_score",
    "balanced_score", "high_pip_score",
]
N_PAIR_FEATURES = len(_PAIR_FEATURE_NAMES)  # 31
N_OPENING_FEATURES = N_VERTEX_FEATURES + N_VERTEX_FEATURES + N_PAIR_FEATURES  # 22+22+31=75

FEATURE_NAMES: list[str] = (
    [f"v1_{n}" for n in [
        "total_pips", "resource_count",
        "has_wood", "has_brick", "has_sheep", "has_wheat", "has_ore",
        "wood_pips", "brick_pips", "sheep_pips", "wheat_pips", "ore_pips",
        "food_pips", "city_pips", "settlement_pips",
        "has_port", "port_ratio", "port_matches_production",
        "robber_vulnerability", "num_adjacent_hexes", "num_adjacent_vertices",
        "resource_entropy",
    ]]
    + [f"v2_{n}" for n in [
        "total_pips", "resource_count",
        "has_wood", "has_brick", "has_sheep", "has_wheat", "has_ore",
        "wood_pips", "brick_pips", "sheep_pips", "wheat_pips", "ore_pips",
        "food_pips", "city_pips", "settlement_pips",
        "has_port", "port_ratio", "port_matches_production",
        "robber_vulnerability", "num_adjacent_hexes", "num_adjacent_vertices",
        "resource_entropy",
    ]]
    + _PAIR_FEATURE_NAMES
)


def opening_features_to_array(of: OpeningFeatures) -> np.ndarray:
    """Fixed-length float32 array: [v1 (22), v2 (22), pair-level (31)] = 75 features."""
    pair = np.array([
        of.combined_pip_count,
        of.unique_resource_count,
        float(of.has_all_five),
        of.settlement_resource_count,
        of.city_resource_count,
        of.combined_wood_pips,
        of.combined_brick_pips,
        of.combined_sheep_pips,
        of.combined_wheat_pips,
        of.combined_ore_pips,
        of.combined_food_pips,
        of.combined_city_pips,
        of.combined_settlement_pips,
        of.num_ports,
        float(of.has_specific_port),
        of.best_port_ratio,
        of.port_synergy_score,
        of.vertex_road_distance,
        of.expansion_vertex_count,
        of.expansion_pip_sum,
        of.expansion_pip_mean,
        of.shared_expansion_hexes,
        of.combined_robber_vulnerability,
        of.double_exposure_count,
        of.seat,
        float(of.is_early_seat),
        of.ore_wheat_score,
        of.road_race_score,
        of.port_engine_score,
        of.balanced_score,
        of.high_pip_score,
    ], dtype=np.float32)
    return np.concatenate([of.v1_arr, of.v2_arr, pair])


def identify_archetype(of: OpeningFeatures) -> str:
    """Rule-based archetype assignment."""
    scores = {
        "ore_wheat": of.ore_wheat_score,
        "road_race": of.road_race_score,
        "port_engine": of.port_engine_score,
        "balanced": of.balanced_score,
        "high_pip": of.high_pip_score,
    }
    # Override: port_engine requires actual port
    if of.num_ports == 0:
        scores["port_engine"] = -1.0
    return max(scores, key=lambda k: scores[k])
