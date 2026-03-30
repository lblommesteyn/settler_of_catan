"""Tests for feature engineering."""

import numpy as np
import pytest

from catan.board.board import CatanBoard, Resource
from catan.features.vertex_features import (
    compute_vertex_features,
    compute_all_vertex_features,
    vertex_features_to_array,
    N_VERTEX_FEATURES,
)
from catan.features.opening_features import (
    compute_opening_features,
    opening_features_to_array,
    N_OPENING_FEATURES,
    identify_archetype,
    ARCHETYPE_NAMES,
    FEATURE_NAMES,
)


@pytest.fixture
def board():
    return CatanBoard.random(seed=42)


# ---------------------------------------------------------------------------
# Vertex features
# ---------------------------------------------------------------------------

def test_vertex_features_all_vertices(board):
    vf_cache = compute_all_vertex_features(board)
    assert len(vf_cache) == 54


def test_vertex_feature_array_shape(board):
    vf_cache = compute_all_vertex_features(board)
    vk = next(iter(vf_cache))
    arr = vertex_features_to_array(vf_cache[vk])
    assert arr.shape == (N_VERTEX_FEATURES,)
    assert arr.dtype == np.float32


def test_vertex_pip_count_non_negative(board):
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        assert vf.total_pips >= 0


def test_vertex_resource_count_range(board):
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        assert 0 <= vf.resource_count <= 3


def test_vertex_pip_consistency(board):
    """total_pips should equal sum of individual resource pips."""
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        total = (
            vf.wood_pips + vf.brick_pips + vf.sheep_pips
            + vf.wheat_pips + vf.ore_pips
        )
        assert vf.total_pips == total


def test_vertex_food_city_settlement_pips(board):
    """Derived pip sums should match individual resource pips."""
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        assert vf.food_pips == vf.wheat_pips + vf.sheep_pips
        assert vf.city_pips == vf.ore_pips + vf.wheat_pips
        assert vf.settlement_pips == (
            vf.wood_pips + vf.brick_pips + vf.sheep_pips + vf.wheat_pips
        )


def test_vertex_resource_entropy_range(board):
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        assert 0.0 <= vf.resource_entropy <= 1.0 + 1e-6


def test_vertex_port_ratio_valid(board):
    for vk in board.all_vertices():
        vf = compute_vertex_features(vk, board)
        assert vf.port_ratio in (0.0, 1 / 3, 1 / 2)


# ---------------------------------------------------------------------------
# Opening features
# ---------------------------------------------------------------------------

def test_opening_features_shape(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2_candidates = board.legal_second_vertices(v1)
    assert len(v2_candidates) > 0
    v2 = v2_candidates[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    arr = opening_features_to_array(of)
    assert arr.shape == (N_OPENING_FEATURES,)
    assert arr.dtype == np.float32


def test_opening_combined_pip_count(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    f1 = vf_cache[v1]
    f2 = vf_cache[v2]
    assert of.combined_pip_count == f1.total_pips + f2.total_pips


def test_opening_unique_resource_count(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    assert 0 <= of.unique_resource_count <= 5


def test_opening_expansion_non_negative(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    assert of.expansion_vertex_count >= 0
    assert of.expansion_pip_sum >= 0


def test_opening_seat_encoding(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    for seat in range(4):
        of = compute_opening_features(v1, v2, seat=seat, board=board, vf_cache=vf_cache)
        assert of.seat == seat
        assert of.is_early_seat == (seat in (0, 1))


def test_opening_num_ports_range(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    assert of.num_ports in (0, 1, 2)


def test_identify_archetype_valid(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    archetype = identify_archetype(of)
    assert archetype in ARCHETYPE_NAMES


def test_feature_names_length():
    assert len(FEATURE_NAMES) == N_OPENING_FEATURES


def test_feature_array_no_nan(board):
    vf_cache = compute_all_vertex_features(board)
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    of = compute_opening_features(v1, v2, seat=0, board=board, vf_cache=vf_cache)
    arr = opening_features_to_array(of)
    assert not np.any(np.isnan(arr)), "Feature array contains NaN"
    assert not np.any(np.isinf(arr)), "Feature array contains Inf"
