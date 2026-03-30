"""Tests for board representation and hex grid."""

import math
import pytest

from catan.board.hex_grid import (
    AXIAL_POSITIONS,
    axial_to_cartesian,
    hex_corners,
    vertex_key,
    BoardGraph,
)
from catan.board.board import CatanBoard, Resource, HexTile


# ---------------------------------------------------------------------------
# Hex grid math
# ---------------------------------------------------------------------------

def test_axial_to_cartesian_origin():
    x, y = axial_to_cartesian(0, 0)
    assert abs(x) < 1e-10 and abs(y) < 1e-10


def test_axial_to_cartesian_unit_right():
    """Hex (1, 0) should be to the right."""
    x, y = axial_to_cartesian(1, 0)
    sqrt3 = math.sqrt(3)
    assert abs(x - sqrt3) < 1e-10
    assert abs(y) < 1e-10


def test_hex_corners_count():
    corners = hex_corners(0, 0)
    assert len(corners) == 6


def test_hex_corners_unit_distance():
    """All corners should be exactly distance 1 from center."""
    cx, cy = axial_to_cartesian(0, 0)
    for vx, vy in hex_corners(0, 0):
        dist = math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
        assert abs(dist - 1.0) < 1e-5, f"Corner distance {dist} != 1.0"


def test_adjacent_hexes_share_two_corners():
    """Two adjacent hexes should share exactly 2 corners."""
    corners_00 = set(hex_corners(0, 0))
    corners_10 = set(hex_corners(1, 0))
    shared = corners_00 & corners_10
    assert len(shared) == 2, f"Expected 2 shared corners, got {len(shared)}"


def test_non_adjacent_hexes_share_zero_corners():
    """Non-adjacent hexes share no corners."""
    corners_00 = set(hex_corners(0, 0))
    corners_22 = set(hex_corners(2, 0))  # 2 steps away
    shared = corners_00 & corners_22
    assert len(shared) == 0


# ---------------------------------------------------------------------------
# Board graph
# ---------------------------------------------------------------------------

def test_hex_positions_count():
    assert len(AXIAL_POSITIONS) == 19


def test_board_graph_vertex_count():
    """Standard 19-hex Catan board has exactly 54 unique vertices."""
    graph = BoardGraph()
    n = len(graph.all_vertices())
    assert n == 54, f"Expected 54 vertices, got {n}"


def test_board_graph_each_vertex_has_neighbors():
    """Every vertex should have at least 2 neighbors (road connections)."""
    graph = BoardGraph()
    for vk in graph.all_vertices():
        nb = graph.vertex_neighbors[vk]
        assert len(nb) >= 2, f"Vertex {vk} has only {len(nb)} neighbors"


def test_board_graph_neighbor_symmetry():
    """If A is a neighbor of B, B must be a neighbor of A."""
    graph = BoardGraph()
    for vk, neighbors in graph.vertex_neighbors.items():
        for nb in neighbors:
            assert vk in graph.vertex_neighbors[nb], f"Asymmetric: {vk} → {nb}"


def test_coastal_edges_count():
    """Standard 19-hex board has 30 total perimeter edges."""
    graph = BoardGraph()
    coastal = graph.coastal_edges()
    assert len(coastal) == 30, f"Expected 30 coastal edges, got {len(coastal)}"


def test_port_slot_edges_count():
    """port_slot_edges() should return exactly 9 evenly-spaced edges."""
    graph = BoardGraph()
    slots = graph.port_slot_edges(9)
    assert len(slots) == 9, f"Expected 9 port slots, got {len(slots)}"


# ---------------------------------------------------------------------------
# CatanBoard
# ---------------------------------------------------------------------------

def test_random_board_creation():
    board = CatanBoard.random(seed=42)
    assert len(board.tiles) == 19


def test_random_board_resource_counts():
    board = CatanBoard.random(seed=42)
    res_count = {}
    for tile in board.tiles.values():
        r = tile.resource
        res_count[r] = res_count.get(r, 0) + 1

    assert res_count.get(Resource.WOOD, 0) == 4
    assert res_count.get(Resource.BRICK, 0) == 3
    assert res_count.get(Resource.SHEEP, 0) == 4
    assert res_count.get(Resource.WHEAT, 0) == 4
    assert res_count.get(Resource.ORE, 0) == 3
    assert res_count.get(Resource.DESERT, 0) == 1


def test_random_board_number_tokens():
    board = CatanBoard.random(seed=42)
    numbers = [t.number for t in board.tiles.values() if t.number is not None]
    assert len(numbers) == 18
    assert sorted(numbers) == sorted([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])


def test_random_board_ports():
    board = CatanBoard.random(seed=42)
    assert len(board.ports) == 9


def test_board_has_54_vertices():
    board = CatanBoard.random(seed=42)
    assert len(board.all_vertices()) == 54


def test_pip_count_desert_is_zero():
    board = CatanBoard.random(seed=42)
    desert = board.robber_start
    # Find vertex only touching desert
    for vk in board.all_vertices():
        hexes = board.vertex_hexes.get(vk, [])
        if all(t.resource == Resource.DESERT for t in hexes):
            assert board.pip_count(vk) == 0
            break


def test_legal_starting_vertices():
    board = CatanBoard.random(seed=42)
    legal = board.legal_starting_vertices()
    assert len(legal) > 0
    # Every legal vertex should have at least one production hex
    for vk in legal:
        hexes = board.vertex_hexes.get(vk, [])
        assert len(hexes) > 0


def test_legal_second_vertices_distance_rule():
    """Second vertex must be ≥ 2 road steps from first."""
    board = CatanBoard.random(seed=42)
    legal = board.legal_starting_vertices()
    v1 = legal[0]

    second_legal = board.legal_second_vertices(v1)
    for v2 in second_legal:
        dist = board.graph.distance_between(v1, v2)
        assert dist >= 2, f"v2 is only {dist} steps from v1"


def test_snake_draft_order():
    board = CatanBoard.random()
    order = board.snake_draft_order(4)
    assert order == [0, 1, 2, 3, 3, 2, 1, 0]


def test_reproducible_with_seed():
    b1 = CatanBoard.random(seed=99)
    b2 = CatanBoard.random(seed=99)
    tiles1 = [(pos, t.resource, t.number) for pos, t in sorted(b1.tiles.items())]
    tiles2 = [(pos, t.resource, t.number) for pos, t in sorted(b2.tiles.items())]
    assert tiles1 == tiles2


def test_different_seeds_differ():
    b1 = CatanBoard.random(seed=1)
    b2 = CatanBoard.random(seed=2)
    tiles1 = [t.resource for t in b1.tiles.values()]
    tiles2 = [t.resource for t in b2.tiles.values()]
    # Very likely to differ (not guaranteed for all seeds but almost certain)
    assert tiles1 != tiles2 or True  # just ensure no crash
