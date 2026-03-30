"""
Board → PyG graph conversion.

Every Catan game has the same 19-hex topology (same 54 vertices, same adjacency).
What varies per game: node features (which resources / pips / ports each vertex has)
and task inputs (which vertices are v1/v2, which seat is playing).

Node features  (NODE_DIM = 16):
  0   wood_pips    / 10          — production weighted by resource type
  1   brick_pips   / 10
  2   sheep_pips   / 10
  3   wheat_pips   / 10
  4   ore_pips     / 10
  5   total_pips   / 30          — overall production
  6   num_adj_hexes / 3          — interior (3) vs coastal (1-2)
  7   has_port                   — binary
  8   port_generic               — one-hot port type (7 options incl. none)
  9   port_wood
  10  port_brick
  11  port_sheep
  12  port_wheat
  13  port_ore
  14  is_v1                      — task input: first settlement here?
  15  is_v2                      — task input: second settlement here?

Graph-level (global) feature (4):
  seat one-hot (0-3)

Edge index: fixed for all boards (undirected, vertex adjacency in hex grid).
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from catan.board.board import CatanBoard, Resource, PortType
from catan.board.hex_grid import BoardGraph

NODE_DIM  = 16
SEAT_DIM  = 4

PORT_TYPE_INDEX = {
    None:              0,   # no port → slot 8 stays 0, rest 0
    PortType.GENERIC:  0,   # generic → port_generic=1
    PortType.WOOD:     1,
    PortType.BRICK:    2,
    PortType.SHEEP:    3,
    PortType.WHEAT:    4,
    PortType.ORE:      5,
}

# Cache the fixed edge_index (topology never changes)
_CACHED_EDGE_INDEX: torch.Tensor | None = None


def _get_edge_index(board: CatanBoard) -> torch.Tensor:
    global _CACHED_EDGE_INDEX
    if _CACHED_EDGE_INDEX is not None:
        return _CACHED_EDGE_INDEX

    verts = _sorted_vertices(board)
    v_idx = {v: i for i, v in enumerate(verts)}

    src, dst = [], []
    for v in verts:
        for nb in board.graph.vertex_neighbors[v]:
            if nb in v_idx:
                src.append(v_idx[v])
                dst.append(v_idx[nb])

    ei = torch.tensor([src, dst], dtype=torch.long)
    _CACHED_EDGE_INDEX = ei
    return ei


def _sorted_vertices(board: CatanBoard) -> list:
    """Consistent vertex ordering: sort by (y, x) rounded to 6dp."""
    return sorted(board.graph.vertices.keys(),
                  key=lambda v: (round(v[1], 6), round(v[0], 6)))


def board_to_node_features(board: CatanBoard) -> np.ndarray:
    """
    Returns float32 array of shape (54, NODE_DIM) with raw board features.
    Does NOT include is_v1 / is_v2 (those are added per-opening).
    """
    verts = _sorted_vertices(board)
    X = np.zeros((len(verts), NODE_DIM), dtype=np.float32)

    # Port lookup: vertex → PortType
    vertex_port: dict = {}
    for port in board.ports:
        for v in port.vertices:
            vertex_port[v] = port.port_type

    for i, v in enumerate(verts):
        hexes = board.vertex_hexes.get(v, [])
        wp = bp = sp = wp2 = op = tp = 0.0

        for hx in hexes:
            res = hx.resource
            pip = float(hx.pips)
            if res == Resource.WOOD:    wp  += pip
            elif res == Resource.BRICK: bp  += pip
            elif res == Resource.SHEEP: sp  += pip
            elif res == Resource.WHEAT: wp2 += pip
            elif res == Resource.ORE:   op  += pip
            tp += pip

        X[i, 0] = wp  / 10.0
        X[i, 1] = bp  / 10.0
        X[i, 2] = sp  / 10.0
        X[i, 3] = wp2 / 10.0
        X[i, 4] = op  / 10.0
        X[i, 5] = tp  / 30.0
        X[i, 6] = len(hexes) / 3.0

        ptype = vertex_port.get(v)
        if ptype is not None:
            X[i, 7] = 1.0
            pi = PORT_TYPE_INDEX.get(ptype, -1)
            if pi >= 0:
                X[i, 8 + pi] = 1.0

        # is_v1 / is_v2 left at 0 — filled in per opening

    return X  # (54, 16)


def opening_to_graph(
    board: CatanBoard,
    v1: tuple[float, float],
    v2: tuple[float, float],
    seat: int,
    label_win: float,
    label_rank: float | None = None,
    node_features_base: np.ndarray | None = None,
) -> Data:
    """
    Convert one (board, v1, v2, seat) opening into a PyG Data object.

    node_features_base: precomputed (54, 14) features — saves recomputing per opening.
    If None, computed from board.
    """
    verts = _sorted_vertices(board)
    v_idx = {v: i for i, v in enumerate(verts)}

    if node_features_base is None:
        node_features_base = board_to_node_features(board)

    # Clone base features and set v1/v2 flags
    X = node_features_base.copy()
    if v1 in v_idx:
        X[v_idx[v1], 14] = 1.0
    if v2 in v_idx:
        X[v_idx[v2], 15] = 1.0

    edge_index = _get_edge_index(board)

    # Global feature: seat one-hot
    u = torch.zeros(SEAT_DIM, dtype=torch.float32)
    if 0 <= seat < SEAT_DIM:
        u[seat] = 1.0

    data = Data(
        x          = torch.from_numpy(X),          # (54, 16)
        edge_index = edge_index,                   # (2, E)
        u          = u.unsqueeze(0),               # (1, 4)
        y          = torch.tensor([label_win], dtype=torch.float32),
    )

    if label_rank is not None:
        # Normalise rank 1-4 → 1.0-0.0 (rank 1 = best = 1.0)
        data.y_rank = torch.tensor([(4 - label_rank) / 3.0], dtype=torch.float32)

    return data
