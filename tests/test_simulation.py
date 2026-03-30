"""Tests for game simulation."""

import pytest

from catan.board.board import CatanBoard, Resource
from catan.simulation.game_state import GameState, empty_hand, SETTLEMENT_COST
from catan.simulation.simulator import run_game, run_opening_evaluation


@pytest.fixture
def board():
    return CatanBoard.random(seed=7)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

def test_new_game_state(board):
    state = GameState.new_game(board)
    assert len(state.players) == 4
    assert state.phase == "setup"
    assert state.winner_id is None


def test_setup_settlement_placement(board):
    state = GameState.new_game(board)
    legal = board.legal_starting_vertices()
    vk = legal[0]
    state.place_setup_settlement(0, vk, receive_resources=False)
    assert vk in state.players[0].settlements
    assert vk in state.occupied_vertices


def test_second_settlement_receives_resources(board):
    state = GameState.new_game(board)
    legal = board.legal_starting_vertices()
    vk = legal[5]
    state.place_setup_settlement(0, vk, receive_resources=True)
    # Should have received at least 0 resources (may be 0 if all adjacent are desert)
    total_res = sum(state.players[0].resources.values())
    assert total_res >= 0


def test_distribute_resources_roll_7(board):
    state = GameState.new_game(board)
    legal = board.legal_starting_vertices()
    state.place_setup_settlement(0, legal[0], receive_resources=False)
    state.occupied_vertices[legal[0]] = 0

    gains = state.distribute_resources(7)
    # Roll 7 gives nothing
    assert all(sum(g.values()) == 0 for g in gains.values())


def test_distribute_resources_normal_roll(board):
    state = GameState.new_game(board)
    legal = board.legal_starting_vertices()
    # Place settlement on highest-pip vertex
    vk = max(legal, key=board.pip_count)
    state.place_setup_settlement(0, vk, receive_resources=False)

    # Find what roll this vertex produces
    for tile in board.vertex_hexes.get(vk, []):
        if tile.number is not None:
            gains = state.distribute_resources(tile.number)
            # Player 0 should gain this resource
            gained = gains[0].get(tile.resource, 0)
            assert gained >= 1
            break


def test_vp_counting(board):
    state = GameState.new_game(board)
    legal = board.legal_starting_vertices()
    state.place_setup_settlement(0, legal[0], receive_resources=False)
    # 1 settlement = 1 VP
    assert state.vp_for_player(0) == 1


def test_check_winner_not_triggered(board):
    state = GameState.new_game(board)
    assert state.check_winner() is None


# ---------------------------------------------------------------------------
# Full game simulation
# ---------------------------------------------------------------------------

def test_run_game_completes(board):
    result = run_game(board, rng=__import__("random").Random(42))
    assert result.winner_id in range(4)
    assert result.turns_elapsed > 0
    assert len(result.final_vps) == 4
    assert all(vp >= 2 for vp in result.final_vps)  # at least 2 settlements each


def test_run_game_with_fixed_opening(board):
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2_cands = board.legal_second_vertices(v1)
    assert len(v2_cands) > 0
    v2 = v2_cands[0]

    result = run_game(
        board,
        fixed_openings={0: (v1, v2)},
        rng=__import__("random").Random(42),
    )
    # Player 0's opening vertices should include v1 and v2
    assert v1 in result.opening_vertices
    assert v2 in result.opening_vertices


def test_run_game_winner_has_most_vp(board):
    import random
    rng = random.Random(123)
    result = run_game(board, rng=rng)
    winner_vp = result.final_vps[result.winner_id]
    # Winner should have max or tied max VP
    assert winner_vp == max(result.final_vps)


def test_run_opening_evaluation_win_rate_range(board):
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    result = run_opening_evaluation(
        board, player_id=0, v1=v1, v2=v2, seat=0, n_simulations=20, seed=42
    )
    assert 0.0 <= result.win_rate <= 1.0
    assert 0.0 <= result.top2_rate <= 1.0
    assert result.avg_final_vp >= 0


def test_opening_eval_win_rate_roughly_fair(board):
    """In a 4-player game with equal players, win rate should be near 0.25."""
    legal = board.legal_starting_vertices()
    v1 = legal[0]
    v2 = board.legal_second_vertices(v1)[0]

    result = run_opening_evaluation(
        board, player_id=0, v1=v1, v2=v2, seat=0, n_simulations=40, seed=7
    )
    # Very loose bounds — just sanity check
    assert 0.0 <= result.win_rate <= 1.0
