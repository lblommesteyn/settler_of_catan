"""Tests for the exact-rule foundation used by the full solver."""

from __future__ import annotations

from dataclasses import replace

from catan.board.board import CatanBoard, PortType, Resource
from catan.full_solver import (
    DEV_CARD_DISTRIBUTION,
    MaritimeTrade,
    TradeOffer,
    TurnPhase,
    accept_pending_trade,
    apply_maritime_trade,
    exact_longest_road_length,
    legal_maritime_trades,
    make_bank_resources,
    make_dev_deck,
    make_initial_state,
    refresh_public_state,
    resolve_resource_shortage,
    start_domestic_trade,
    total_victory_points,
    validate_trade_offer,
)


def _path_edges(graph, length: int) -> list[frozenset]:
    def dfs(vertex, path_vertices, path_edges):
        if len(path_edges) == length:
            return list(path_edges)
        for neighbor in graph.vertex_neighbors[vertex]:
            edge = frozenset({vertex, neighbor})
            if edge in path_edges or neighbor in path_vertices:
                continue
            path_vertices.add(neighbor)
            path_edges.append(edge)
            result = dfs(neighbor, path_vertices, path_edges)
            if result is not None:
                return result
            path_edges.pop()
            path_vertices.remove(neighbor)
        return None

    for start_vertex in graph.all_vertices():
        result = dfs(start_vertex, {start_vertex}, [])
        if result is not None:
            return result
    raise RuntimeError(f"Unable to find a simple path of length {length}")


def test_dev_deck_matches_base_game_distribution():
    deck = make_dev_deck(seed=7)
    assert len(deck) == 25
    for card_type, expected in DEV_CARD_DISTRIBUTION.items():
        assert deck.count(card_type) == expected


def test_trade_offer_rejects_gifts_and_like_resource_swaps():
    offer = TradeOffer(
        offerer=0,
        responder=1,
        give={Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
        receive={Resource.WOOD: 1, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
    )
    errors = validate_trade_offer(offer)
    assert any("give at least one resource" in error.lower() for error in errors)

    like_for_like = TradeOffer(
        offerer=0,
        responder=1,
        give={Resource.WOOD: 2, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
        receive={Resource.WOOD: 1, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
    )
    like_errors = validate_trade_offer(like_for_like)
    assert any("like resources" in error.lower() for error in like_errors)


def test_domestic_trade_updates_private_hands_and_public_hand_sizes():
    board = CatanBoard.random(seed=11)
    state = make_initial_state(board)
    private_players = list(state.private_players)
    private_players[0] = replace(
        private_players[0],
        resources={
            Resource.WOOD: 1,
            Resource.BRICK: 1,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        },
    )
    private_players[1] = replace(
        private_players[1],
        resources={
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 1,
            Resource.ORE: 0,
        },
    )
    state = replace(
        state,
        private_players=tuple(private_players),
        phase=TurnPhase.MAIN,
        current_player=0,
    )
    state = refresh_public_state(state)

    offer = TradeOffer(
        offerer=0,
        responder=1,
        give={Resource.WOOD: 1, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
        receive={Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 1, Resource.ORE: 0},
    )
    state = start_domestic_trade(state, offer)
    state = accept_pending_trade(state, responder_id=1)

    assert state.private_players[0].resources[Resource.WHEAT] == 1
    assert state.private_players[0].resources[Resource.WOOD] == 0
    assert state.private_players[1].resources[Resource.WHEAT] == 0
    assert state.private_players[1].resources[Resource.WOOD] == 1
    assert state.public_players[0].hand_size == 2
    assert state.public_players[1].hand_size == 1


def test_legal_maritime_trades_honor_best_port_ratio():
    board = CatanBoard.random(seed=5)
    state = make_initial_state(board)
    private_players = list(state.private_players)
    public_players = list(state.public_players)

    private_players[0] = replace(
        private_players[0],
        resources={
            Resource.WOOD: 2,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        },
    )
    public_players[0] = replace(public_players[0], ports=(PortType.WOOD,))

    state = replace(
        state,
        private_players=tuple(private_players),
        public_players=tuple(public_players),
        phase=TurnPhase.MAIN,
        current_player=0,
    )

    trades = legal_maritime_trades(state, 0)
    assert trades
    assert all(trade.give_count == 2 for trade in trades)
    assert all(trade.give_resource == Resource.WOOD for trade in trades)


def test_apply_maritime_trade_updates_bank_and_player_resources():
    board = CatanBoard.random(seed=17)
    state = make_initial_state(board)
    private_players = list(state.private_players)
    public_players = list(state.public_players)
    private_players[0] = replace(
        private_players[0],
        resources={
            Resource.WOOD: 4,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        },
    )
    public_players[0] = replace(public_players[0], ports=())

    state = replace(
        state,
        private_players=tuple(private_players),
        public_players=tuple(public_players),
        phase=TurnPhase.MAIN,
        current_player=0,
    )
    trade = MaritimeTrade(
        player_id=0,
        give_resource=Resource.WOOD,
        give_count=4,
        receive_resource=Resource.ORE,
    )
    state = apply_maritime_trade(state, trade)
    assert state.private_players[0].resources[Resource.WOOD] == 0
    assert state.private_players[0].resources[Resource.ORE] == 1
    assert state.bank_resources[Resource.WOOD] == make_bank_resources()[Resource.WOOD] + 4
    assert state.bank_resources[Resource.ORE] == make_bank_resources()[Resource.ORE] - 1


def test_resource_shortage_single_recipient_gets_all_available():
    bank = make_bank_resources()
    bank[Resource.ORE] = 1
    payouts, updated_bank = resolve_resource_shortage(
        bank,
        {
            0: {
                Resource.WOOD: 0,
                Resource.BRICK: 0,
                Resource.SHEEP: 0,
                Resource.WHEAT: 0,
                Resource.ORE: 2,
            }
        },
    )
    assert payouts[0][Resource.ORE] == 1
    assert updated_bank[Resource.ORE] == 0


def test_resource_shortage_multiple_recipients_get_nothing():
    bank = make_bank_resources()
    bank[Resource.WHEAT] = 1
    payouts, updated_bank = resolve_resource_shortage(
        bank,
        {
            0: {
                Resource.WOOD: 0,
                Resource.BRICK: 0,
                Resource.SHEEP: 0,
                Resource.WHEAT: 1,
                Resource.ORE: 0,
            },
            1: {
                Resource.WOOD: 0,
                Resource.BRICK: 0,
                Resource.SHEEP: 0,
                Resource.WHEAT: 1,
                Resource.ORE: 0,
            },
        },
    )
    assert payouts[0][Resource.WHEAT] == 0
    assert payouts[1][Resource.WHEAT] == 0
    assert updated_bank[Resource.WHEAT] == 1


def test_exact_longest_road_counts_cycle_of_six():
    board = CatanBoard.random(seed=21)
    cycle_vertices = board.graph.hex_vertices[(0, 0)]
    cycle_edges = [frozenset({cycle_vertices[i], cycle_vertices[(i + 1) % 6]}) for i in range(6)]
    assert exact_longest_road_length(board, frozenset(cycle_edges)) == 6


def test_exact_longest_road_uses_longest_branch_not_total_edges():
    board = CatanBoard.random(seed=22)
    center = next(vertex for vertex, neighbors in board.graph.vertex_neighbors.items() if len(neighbors) >= 3)
    neighbors = board.graph.vertex_neighbors[center]
    branch_edges = [
        frozenset({center, neighbors[0]}),
        frozenset({center, neighbors[1]}),
        frozenset({center, neighbors[2]}),
    ]
    next_hop = next(vertex for vertex in board.graph.vertex_neighbors[neighbors[2]] if vertex != center)
    branch_edges.append(frozenset({neighbors[2], next_hop}))

    assert exact_longest_road_length(board, frozenset(branch_edges)) == 3


def test_exact_longest_road_is_interrupted_by_opponent_settlement():
    board = CatanBoard.random(seed=23)
    path_edges = _path_edges(board.graph, length=4)
    blocked_vertex = next(iter(path_edges[1] & path_edges[2]))
    assert exact_longest_road_length(board, frozenset(path_edges), {blocked_vertex}) == 2


def test_refresh_public_state_and_total_vp_include_hidden_vps_and_longest_road():
    board = CatanBoard.random(seed=24)
    state = make_initial_state(board)
    path_edges = _path_edges(board.graph, length=5)
    path_vertices = set()
    for edge in path_edges:
        path_vertices.update(edge)

    public_players = list(state.public_players)
    private_players = list(state.private_players)
    public_players[0] = replace(
        public_players[0],
        settlements=frozenset(set(list(path_vertices)[:2])),
        roads=frozenset(path_edges),
    )
    private_players[0] = replace(private_players[0], hidden_vp_cards=2)
    state = replace(state, public_players=tuple(public_players), private_players=tuple(private_players))
    state = refresh_public_state(state)

    assert state.public_players[0].has_longest_road
    assert state.public_players[0].visible_vp == 4
    assert total_victory_points(state, 0) == 6
