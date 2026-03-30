"""
Per-vertex feature computation.

A vertex's features describe its production potential, resource mix,
port access, and robber vulnerability — all from the board alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..board.board import CatanBoard, Resource, PortType, PIP_VALUES


PRODUCTION_RESOURCES = {Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT}
CITY_RESOURCES = {Resource.ORE, Resource.WHEAT}
SETTLEMENT_RESOURCES = {Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT}


@dataclass
class VertexFeatures:
    vertex: tuple[float, float]

    # --- Production ---
    total_pips: int                  # sum of pip dots on adjacent number hexes
    resource_count: int              # distinct non-desert resources (1–3)
    has_wood: bool
    has_brick: bool
    has_sheep: bool
    has_wheat: bool
    has_ore: bool
    wood_pips: int
    brick_pips: int
    sheep_pips: int
    wheat_pips: int
    ore_pips: int
    food_pips: int                   # wheat + sheep  (city/dev path)
    city_pips: int                   # ore + wheat
    settlement_pips: int             # wood + brick + sheep + wheat

    # --- Port ---
    has_port: bool
    port_type: Optional[PortType]
    port_ratio: float                # 1/3 generic, 1/2 resource-specific, 0 none
    port_matches_production: bool    # 2:1 port resource is actually produced here

    # --- Robber exposure ---
    robber_vulnerability: float      # sum of P(roll) for adjacent number hexes / #adjacent
    # high-pip hexes are more likely to be targeted

    # --- Topology ---
    num_adjacent_hexes: int          # 1–3
    num_adjacent_vertices: int       # degree in road graph

    # --- Misc ---
    resource_entropy: float          # Shannon entropy of pip distribution across resources


def compute_vertex_features(
    vk: tuple[float, float],
    board: CatanBoard,
) -> VertexFeatures:
    tiles = board.vertex_hexes.get(vk, [])
    port = board.get_port(vk)
    neighbors = board.graph.vertex_neighbors.get(vk, [])

    # Pip breakdown per resource
    pip_by_res: dict[Resource, int] = {r: 0 for r in Resource if r != Resource.DESERT}
    for tile in tiles:
        if tile.resource != Resource.DESERT and tile.number is not None:
            pip_by_res[tile.resource] = pip_by_res.get(tile.resource, 0) + tile.pips

    total_pips = sum(pip_by_res.values())
    resource_count = sum(1 for r, p in pip_by_res.items() if p > 0)

    # Robber vulnerability: sum of dice probability for each adjacent production hex
    # P(roll=n) = PIP_VALUES[n] / 36
    rob_prob = 0.0
    n_prod_hexes = 0
    for tile in tiles:
        if tile.number is not None:
            rob_prob += PIP_VALUES[tile.number] / 36.0
            n_prod_hexes += 1

    # Port attributes
    if port is not None:
        pt = port.port_type
        has_port = True
        port_ratio = port.ratio
        if port.resource is not None:
            matches = pip_by_res.get(port.resource, 0) > 0
        else:
            matches = total_pips > 0
    else:
        pt = None
        has_port = False
        port_ratio = 0.0
        matches = False

    # Resource entropy (diversity measure), normalized to [0, 1] using log2(5) as max
    total = total_pips if total_pips > 0 else 1
    probs = [p / total for p in pip_by_res.values() if p > 0]
    raw_entropy = -sum(p * math.log2(p) for p in probs) if probs else 0.0
    entropy = raw_entropy / math.log2(5) if raw_entropy > 0 else 0.0  # max 5 resources

    return VertexFeatures(
        vertex=vk,
        total_pips=total_pips,
        resource_count=resource_count,
        has_wood=pip_by_res[Resource.WOOD] > 0,
        has_brick=pip_by_res[Resource.BRICK] > 0,
        has_sheep=pip_by_res[Resource.SHEEP] > 0,
        has_wheat=pip_by_res[Resource.WHEAT] > 0,
        has_ore=pip_by_res[Resource.ORE] > 0,
        wood_pips=pip_by_res[Resource.WOOD],
        brick_pips=pip_by_res[Resource.BRICK],
        sheep_pips=pip_by_res[Resource.SHEEP],
        wheat_pips=pip_by_res[Resource.WHEAT],
        ore_pips=pip_by_res[Resource.ORE],
        food_pips=pip_by_res[Resource.WHEAT] + pip_by_res[Resource.SHEEP],
        city_pips=pip_by_res[Resource.ORE] + pip_by_res[Resource.WHEAT],
        settlement_pips=sum(
            pip_by_res[r] for r in SETTLEMENT_RESOURCES
        ),
        has_port=has_port,
        port_type=pt,
        port_ratio=port_ratio,
        port_matches_production=matches,
        robber_vulnerability=rob_prob,
        num_adjacent_hexes=len(tiles),
        num_adjacent_vertices=len(neighbors),
        resource_entropy=entropy,
    )


def compute_all_vertex_features(
    board: CatanBoard,
) -> dict[tuple[float, float], VertexFeatures]:
    """Compute and cache features for every vertex. Call once per board."""
    return {vk: compute_vertex_features(vk, board) for vk in board.all_vertices()}


# Feature vector ordering (must stay stable for ML models)
_FEATURE_NAMES = [
    "total_pips", "resource_count",
    "has_wood", "has_brick", "has_sheep", "has_wheat", "has_ore",
    "wood_pips", "brick_pips", "sheep_pips", "wheat_pips", "ore_pips",
    "food_pips", "city_pips", "settlement_pips",
    "has_port", "port_ratio", "port_matches_production",
    "robber_vulnerability",
    "num_adjacent_hexes", "num_adjacent_vertices",
    "resource_entropy",
]
N_VERTEX_FEATURES = len(_FEATURE_NAMES)  # 22


def vertex_features_to_array(vf: VertexFeatures) -> np.ndarray:
    """Convert to a fixed-length float32 array."""
    return np.array([
        vf.total_pips,
        vf.resource_count,
        float(vf.has_wood),
        float(vf.has_brick),
        float(vf.has_sheep),
        float(vf.has_wheat),
        float(vf.has_ore),
        vf.wood_pips,
        vf.brick_pips,
        vf.sheep_pips,
        vf.wheat_pips,
        vf.ore_pips,
        vf.food_pips,
        vf.city_pips,
        vf.settlement_pips,
        float(vf.has_port),
        vf.port_ratio,
        float(vf.port_matches_production),
        vf.robber_vulnerability,
        vf.num_adjacent_hexes,
        vf.num_adjacent_vertices,
        vf.resource_entropy,
    ], dtype=np.float32)
