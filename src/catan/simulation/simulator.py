"""
Catan game simulator.

Runs simplified 4-player Catan games to estimate opening quality.
The GreedyBot is intentionally simple — consistent labels > realistic play.

Key function: run_opening_evaluation()
  Given a board and a fixed (v1, v2) opening for one player,
  simulate N games with greedy bots for all players and estimate win rate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..board.board import CatanBoard, Resource, PIP_VALUES
from .game_state import (
    GameState, PlayerState, ROAD_COST, SETTLEMENT_COST, CITY_COST,
    empty_hand,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    winner_id: int
    final_vps: list[int]          # VP per player (index = player_id)
    turns_elapsed: int
    vp_trajectory: dict[int, list[int]]  # {player_id: [vp at each turn]}
    first_city_turn: dict[int, Optional[int]]
    opening_vertices: list[tuple[float, float]]  # all opening settlements in draft order


@dataclass
class OpeningEvalResult:
    player_id: int
    v1: tuple[float, float]
    v2: tuple[float, float]
    seat: int
    n_simulations: int
    win_rate: float
    top2_rate: float
    avg_final_vp: float
    avg_vp_at_turn_30: float
    avg_first_city_turn: float   # NaN if never achieved
    raw_win_count: int


# ---------------------------------------------------------------------------
# Greedy bot
# ---------------------------------------------------------------------------

class GreedyBot:
    """
    Deterministic greedy policy:
      - Opening: pick highest pip-count legal vertex
      - Main game: build in priority order: city > settlement > road > (skip)
    """

    def __init__(self, rng: random.Random):
        self.rng = rng

    # --- Opening ---

    def choose_opening_vertex(
        self,
        state: GameState,
        player_id: int,
        occupied: set[tuple[float, float]],
        placement_round: int,  # 0 = first settlement, 1 = second
    ) -> tuple[float, float]:
        """Pick the highest pip-count legal vertex."""
        board = state.board
        if placement_round == 0:
            # Exclude occupied vertices AND their immediate neighbors (distance rule)
            too_close: set[tuple[float, float]] = set()
            for occ in occupied:
                too_close.add(occ)
                too_close.update(board.graph.vertex_neighbors[occ])
            candidates = [v for v in board.legal_starting_vertices() if v not in too_close]
        else:
            first = state.players[player_id].settlements[0]
            candidates = board.legal_second_vertices(first, occupied)

        if not candidates:
            # Fallback: any unoccupied vertex not adjacent to opponent
            candidates = [
                v for v in board.all_vertices()
                if v not in occupied
                and all(nb not in occupied for nb in board.graph.vertex_neighbors[v])
            ]

        # Score by pip count, break ties randomly
        def score(v):
            return (board.pip_count(v), self.rng.random())

        return max(candidates, key=score)

    def choose_opening_road(
        self,
        state: GameState,
        player_id: int,
        settlement_vertex: tuple[float, float],
    ) -> frozenset:
        """Road adjacent to just-placed settlement toward best expansion."""
        board = state.board
        p = state.players[player_id]
        own_verts = set(p.settlements) | set(p.cities)

        # Edges adjacent to this settlement not yet occupied
        candidates = []
        for nb in board.graph.vertex_neighbors[settlement_vertex]:
            edge = frozenset({settlement_vertex, nb})
            if edge not in state.occupied_edges:
                # Score by pip count at the far vertex
                far_pips = board.pip_count(nb)
                candidates.append((far_pips, edge))

        if not candidates:
            # Any edge adjacent to own structures
            for v in own_verts:
                for nb in board.graph.vertex_neighbors[v]:
                    edge = frozenset({v, nb})
                    if edge not in state.occupied_edges:
                        candidates.append((board.pip_count(nb), edge))

        if not candidates:
            # Dummy edge (shouldn't happen on valid board)
            return frozenset({settlement_vertex, settlement_vertex})

        return max(candidates, key=lambda x: (x[0], self.rng.random()))[1]

    # --- Main game actions ---

    def choose_action(self, state: GameState, player_id: int) -> Optional[dict]:
        """
        Return the best available build action, or None if nothing can be built.
        Priority: city > settlement > road
        """
        p = state.players[player_id]
        board = state.board

        # Try city
        if p.can_afford(CITY_COST) and p.settlements:
            best_v = max(p.settlements, key=board.pip_count)
            return {"type": "city", "vertex": best_v}

        # Try settlement
        if p.can_afford(SETTLEMENT_COST) and p.roads:
            road_vertices = {v for r in p.roads for v in tuple(r)}
            candidates = [
                v for v in road_vertices
                if v not in state.occupied_vertices
                and all(nb not in state.occupied_vertices for nb in board.graph.vertex_neighbors[v])
                and board.vertex_hexes.get(v)
            ]
            if candidates:
                best_v = max(candidates, key=lambda v: (board.pip_count(v), self.rng.random()))
                return {"type": "settlement", "vertex": best_v}

        # Try road (only if it opens a new settlement spot)
        if p.can_afford(ROAD_COST):
            own_verts = set(p.settlements) | set(p.cities)
            road_verts = {v for r in p.roads for v in tuple(r)} | own_verts
            # Find edges that extend toward high-pip unoccupied vertices
            candidates = []
            for v in road_verts:
                for nb in board.graph.vertex_neighbors[v]:
                    edge = frozenset({v, nb})
                    if edge not in state.occupied_edges:
                        far_pips = board.pip_count(nb)
                        candidates.append((far_pips, edge))
            if candidates:
                best_edge = max(candidates, key=lambda x: (x[0], self.rng.random()))[1]
                return {"type": "road", "edge": best_edge}

        return None

    def choose_robber_target(
        self, state: GameState, player_id: int
    ) -> tuple[int, int]:
        """Move robber to highest-pip hex of the leading opponent."""
        board = state.board
        # Find leading opponent
        my_vp = state.vp_for_player(player_id)
        opponents = sorted(
            [p for p in state.players if p.player_id != player_id],
            key=lambda p: state.vp_for_player(p.player_id),
            reverse=True,
        )

        if not opponents:
            return board.robber_start

        target_pid = opponents[0].player_id
        target_p = state.players[target_pid]

        # Find hex adjacent to target player's settlements with highest pips
        best_hex = board.robber_start
        best_pips = -1
        for vk in target_p.settlements + target_p.cities:
            for tile in board.vertex_hexes.get(vk, []):
                hk = (tile.q, tile.r)
                if hk != state.robber_hex and tile.pips > best_pips:
                    best_pips = tile.pips
                    best_hex = hk

        return best_hex


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(
    board: CatanBoard,
    fixed_openings: Optional[dict[int, tuple[tuple[float, float], tuple[float, float]]]] = None,
    rng: Optional[random.Random] = None,
    max_turns: int = 300,
) -> SimulationResult:
    """
    Run one complete game.

    fixed_openings: {player_id: (v1, v2)} — pre-set openings for specific players.
    Other players use greedy bot opening selection.
    """
    if rng is None:
        rng = random.Random()

    state = GameState.new_game(board)
    bot = GreedyBot(rng)
    n_players = len(state.players)
    draft_order = board.snake_draft_order(n_players)

    vp_trajectory: dict[int, list[int]] = {i: [] for i in range(n_players)}

    # --- Setup phase ---
    occupied: set[tuple[float, float]] = set()
    opening_vertices: list[tuple[float, float]] = []

    for placement_idx, pid in enumerate(draft_order):
        round_num = 0 if placement_idx < n_players else 1
        is_second = round_num == 1

        if fixed_openings and pid in fixed_openings:
            v1, v2 = fixed_openings[pid]
            vk = v2 if is_second else v1
        else:
            vk = bot.choose_opening_vertex(state, pid, occupied, round_num)

        state.place_setup_settlement(pid, vk, receive_resources=is_second)
        occupied.add(vk)
        opening_vertices.append(vk)

        road = bot.choose_opening_road(state, pid, vk)
        state.place_setup_road(pid, road)

    state.phase = "main"

    # --- Main game loop ---
    for turn in range(max_turns):
        state.current_turn = turn
        pid = turn % n_players

        roll = rng.randint(1, 6) + rng.randint(1, 6)

        if roll == 7:
            # Discard rule (simplified: discard if > 7 cards)
            p = state.players[pid]
            if p.hand_size() > 7:
                excess = p.hand_size() - 7
                _discard(p, excess, rng)
            target = bot.choose_robber_target(state, pid)
            state.move_robber(pid, target)
        else:
            state.distribute_resources(roll)

        # Build actions (may build multiple things per turn)
        for _ in range(4):  # at most 4 builds per turn
            action = bot.choose_action(state, pid)
            if action is None:
                break
            if action["type"] == "city":
                state.build_city(pid, action["vertex"])
            elif action["type"] == "settlement":
                state.build_settlement(pid, action["vertex"])
            elif action["type"] == "road":
                state.build_road(pid, action["edge"])

        # Record VP trajectory
        for p in state.players:
            vp_trajectory[p.player_id].append(state.vp_for_player(p.player_id))

        winner = state.check_winner()
        if winner is not None:
            break

    # Determine winner by VP if no one hit 10
    if state.winner_id is None:
        state.winner_id = max(
            range(n_players),
            key=lambda i: (state.vp_for_player(i), rng.random()),
        )

    return SimulationResult(
        winner_id=state.winner_id,
        final_vps=[state.vp_for_player(i) for i in range(n_players)],
        turns_elapsed=state.current_turn,
        vp_trajectory=vp_trajectory,
        first_city_turn={p.player_id: p.first_city_turn for p in state.players},
        opening_vertices=opening_vertices,
    )


def _discard(p: PlayerState, n: int, rng: random.Random) -> None:
    """Discard n cards from hand (random selection)."""
    cards = [r for r, cnt in p.resources.items() for _ in range(cnt)]
    to_discard = rng.sample(cards, min(n, len(cards)))
    for r in to_discard:
        p.resources[r] = max(0, p.resources[r] - 1)


# ---------------------------------------------------------------------------
# Opening evaluation
# ---------------------------------------------------------------------------

def run_opening_evaluation(
    board: CatanBoard,
    player_id: int,
    v1: tuple[float, float],
    v2: tuple[float, float],
    seat: int,
    n_simulations: int = 100,
    seed: Optional[int] = None,
) -> OpeningEvalResult:
    """
    Estimate win probability for (v1, v2) opening via Monte Carlo.
    player_id is assigned seat position; others use greedy bot.
    """
    rng = random.Random(seed)
    wins = 0
    top2 = 0
    total_vp = 0.0
    vp_at_30: list[float] = []
    city_turns: list[int] = []

    fixed = {player_id: (v1, v2)}

    for _ in range(n_simulations):
        result = run_game(board, fixed_openings=fixed, rng=rng)
        final_vp = result.final_vps[player_id]
        total_vp += final_vp

        if result.winner_id == player_id:
            wins += 1

        # Top-2 rank
        sorted_vps = sorted(result.final_vps, reverse=True)
        if final_vp >= sorted_vps[1]:
            top2 += 1

        # VP at turn 30
        traj = result.vp_trajectory[player_id]
        vp_at_30.append(traj[min(29, len(traj) - 1)] if traj else 0)

        # First city turn
        ct = result.first_city_turn.get(player_id)
        if ct is not None:
            city_turns.append(ct)

    import math
    avg_city = sum(city_turns) / len(city_turns) if city_turns else float("nan")

    return OpeningEvalResult(
        player_id=player_id,
        v1=v1,
        v2=v2,
        seat=seat,
        n_simulations=n_simulations,
        win_rate=wins / n_simulations,
        top2_rate=top2 / n_simulations,
        avg_final_vp=total_vp / n_simulations,
        avg_vp_at_turn_30=sum(vp_at_30) / len(vp_at_30) if vp_at_30 else 0.0,
        avg_first_city_turn=avg_city,
        raw_win_count=wins,
    )


def simulate_dataset(
    n_boards: int = 100,
    openings_per_board: int = 20,
    n_sims_per_opening: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a synthetic training dataset.
    For each random board, sample opening pairs and estimate win probability.
    Returns a list of dicts suitable for pd.DataFrame().
    """
    from ..board.board import CatanBoard as _Board
    from ..features.opening_features import (
        compute_opening_features,
        opening_features_to_array,
        compute_all_vertex_features,
    )

    rng = random.Random(seed)
    rows = []

    for board_idx in range(n_boards):
        board = _Board.random(seed=rng.randint(0, 2**31))
        vf_cache = compute_all_vertex_features(board)
        legal = board.legal_starting_vertices()

        if len(legal) < 2:
            continue

        for _ in range(openings_per_board):
            seat = rng.randint(0, 3)
            pid = seat

            v1 = rng.choice(legal)
            v2_candidates = board.legal_second_vertices(v1)
            if not v2_candidates:
                continue
            v2 = rng.choice(v2_candidates)

            # Compute opening features
            of = compute_opening_features(v1, v2, seat, board, vf_cache)
            arr = opening_features_to_array(of)

            # Evaluate via simulation
            eval_res = run_opening_evaluation(
                board, pid, v1, v2, seat,
                n_simulations=n_sims_per_opening,
                seed=rng.randint(0, 2**31),
            )

            row = {
                "board_idx": board_idx,
                "seat": seat,
                "v1": v1,
                "v2": v2,
                "win_rate": eval_res.win_rate,
                "top2_rate": eval_res.top2_rate,
                "avg_final_vp": eval_res.avg_final_vp,
                "avg_vp_at_30": eval_res.avg_vp_at_turn_30,
                "features": arr,
                # Key opening features for analysis
                "combined_pip_count": of.combined_pip_count,
                "unique_resource_count": of.unique_resource_count,
                "expansion_vertex_count": of.expansion_vertex_count,
                "num_ports": of.num_ports,
                "archetype": __import__(
                    "catan.features.opening_features", fromlist=["identify_archetype"]
                ).identify_archetype(of),
            }
            rows.append(row)

    return rows
