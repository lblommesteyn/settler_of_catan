"""End-to-end tests for the exact full-solver engine."""

from __future__ import annotations

from dataclasses import replace

from catan.board.board import CatanBoard, Resource
from catan.full_solver import (
    Action,
    ActionType,
    DevCardType,
    ExactRulesEngine,
    TradeOffer,
    TurnPhase,
    make_initial_state,
    make_play_knight,
    make_roll,
    refresh_public_state,
)


def _first(actions, action_type: ActionType) -> Action:
    return next(action for action in actions if action.action_type == action_type)


def test_setup_sequence_completes_into_pre_roll():
    board = CatanBoard.random(seed=31)
    engine = ExactRulesEngine()
    state = make_initial_state(board)

    for _ in range(8):
        settlement_action = engine.legal_actions(state)[0]
        assert settlement_action.action_type == ActionType.SETUP_SETTLEMENT
        state = engine.apply_action(state, settlement_action)

        road_action = engine.legal_actions(state)[0]
        assert road_action.action_type == ActionType.SETUP_ROAD
        state = engine.apply_action(state, road_action)

    assert state.phase == TurnPhase.PRE_ROLL
    assert state.current_player == 0
    assert all(len(player.settlements) == 2 for player in state.public_players)
    assert all(len(player.roads) == 2 for player in state.public_players)
    assert sum(state.bank_resources.values()) < 19 * 5


def test_roll_seven_forces_discards_then_robber_move():
    board = CatanBoard.random(seed=32)
    engine = ExactRulesEngine()
    state = make_initial_state(board)

    public_players = list(state.public_players)
    private_players = list(state.private_players)
    victim_vertex = max(board.all_vertices(), key=board.pip_count)
    roller_vertex = next(vertex for vertex in board.all_vertices() if vertex != victim_vertex)
    public_players[0] = replace(public_players[0], settlements=frozenset({roller_vertex}), settlements_left=4)
    public_players[1] = replace(public_players[1], settlements=frozenset({victim_vertex}), settlements_left=4)
    private_players[0] = replace(
        private_players[0],
        resources={resource: 0 for resource in private_players[0].resources},
    )
    private_players[1] = replace(
        private_players[1],
        resources={
            Resource.WOOD: 3,
            Resource.BRICK: 2,
            Resource.SHEEP: 2,
            Resource.WHEAT: 1,
            Resource.ORE: 1,
        },
    )
    state = refresh_public_state(
        replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            current_player=0,
            phase=TurnPhase.PRE_ROLL,
        )
    )

    state = engine.apply_action(state, make_roll(7))
    assert state.phase == TurnPhase.RESOLVE_SEVEN
    assert state.pending_discarders == (1,)

    discard_action = engine.legal_actions(state)[0]
    state = engine.apply_action(state, discard_action)
    assert state.pending_discarders == ()

    robber_actions = engine.legal_actions(state)
    robber_action = next(action for action in robber_actions if action.payload.victim_id == 1)
    prev_roller = state.public_players[0].hand_size
    prev_victim = state.public_players[1].hand_size
    state = engine.apply_action(state, robber_action, seed=7)

    assert state.phase == TurnPhase.MAIN
    assert state.robber_hex == robber_action.payload.target_hex
    assert state.public_players[0].hand_size == prev_roller + 1
    assert state.public_players[1].hand_size == prev_victim - 1


def test_build_road_settlement_and_city_progression():
    board = CatanBoard.random(seed=33)
    engine = ExactRulesEngine()
    state = make_initial_state(board)

    start_vertex = board.all_vertices()[0]
    first_neighbor = board.graph.vertex_neighbors[start_vertex][0]
    first_edge = frozenset({start_vertex, first_neighbor})

    public_players = list(state.public_players)
    private_players = list(state.private_players)
    public_players[0] = replace(
        public_players[0],
        settlements=frozenset({start_vertex}),
        roads=frozenset({first_edge}),
        settlements_left=4,
        roads_left=14,
    )
    private_players[0] = replace(
        private_players[0],
        resources={
            Resource.WOOD: 4,
            Resource.BRICK: 4,
            Resource.SHEEP: 2,
            Resource.WHEAT: 3,
            Resource.ORE: 3,
        },
    )
    state = refresh_public_state(
        replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            current_player=0,
            phase=TurnPhase.MAIN,
            dice_rolled_this_turn=True,
        )
    )

    road_action = _first(engine.legal_actions(state), ActionType.BUILD_ROAD)
    state = engine.apply_action(state, road_action)
    assert state.public_players[0].roads_left == 13

    settlement_action = _first(engine.legal_actions(state), ActionType.BUILD_SETTLEMENT)
    state = engine.apply_action(state, settlement_action)
    assert len(state.public_players[0].settlements) == 2

    city_action = Action(ActionType.BUILD_CITY, payload=settlement_action.payload)
    state = engine.apply_action(state, city_action)
    assert settlement_action.payload in state.public_players[0].cities
    assert settlement_action.payload not in state.public_players[0].settlements


def test_buy_dev_card_stays_fresh_until_end_turn_then_becomes_playable():
    board = CatanBoard.random(seed=34)
    engine = ExactRulesEngine()
    state = make_initial_state(board)

    private_players = list(state.private_players)
    private_players[0] = replace(
        private_players[0],
        resources={
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 1,
            Resource.WHEAT: 1,
            Resource.ORE: 1,
        },
    )
    state = refresh_public_state(
        replace(
            state,
            private_players=tuple(private_players),
            current_player=0,
            phase=TurnPhase.MAIN,
            dice_rolled_this_turn=True,
            dev_deck=(DevCardType.KNIGHT,),
        )
    )

    state = engine.apply_action(state, Action(ActionType.BUY_DEV_CARD))
    assert state.private_players[0].new_dev_cards_in_hand[DevCardType.KNIGHT] == 1
    assert all(action.action_type != ActionType.PLAY_KNIGHT for action in engine.legal_actions(state))

    state = engine.apply_action(state, Action(ActionType.END_TURN))
    assert state.private_players[0].dev_cards_in_hand[DevCardType.KNIGHT] == 1

    state = replace(state, current_player=0, phase=TurnPhase.PRE_ROLL)
    assert any(action.action_type == ActionType.PLAY_KNIGHT for action in engine.legal_actions(state))


def test_knight_claims_largest_army_and_hidden_vp_can_declare_victory():
    board = CatanBoard.random(seed=35)
    engine = ExactRulesEngine()
    state = make_initial_state(board)

    public_players = list(state.public_players)
    private_players = list(state.private_players)
    victim_vertex = max(board.all_vertices(), key=board.pip_count)
    public_players[1] = replace(public_players[1], settlements=frozenset({victim_vertex}), settlements_left=4)
    public_players[0] = replace(public_players[0], settlements=frozenset({board.all_vertices()[0]}), settlements_left=4, played_knights=2)
    private_players[0] = replace(
        private_players[0],
        dev_cards_in_hand={
            DevCardType.KNIGHT: 1,
            DevCardType.ROAD_BUILDING: 0,
            DevCardType.YEAR_OF_PLENTY: 0,
            DevCardType.MONOPOLY: 0,
            DevCardType.VICTORY_POINT: 0,
        },
        hidden_vp_cards=8,
    )
    private_players[1] = replace(
        private_players[1],
        resources={
            Resource.WOOD: 1,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        },
    )
    state = refresh_public_state(
        replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            current_player=0,
            phase=TurnPhase.PRE_ROLL,
        )
    )

    knight_action = next(action for action in engine.legal_actions(state) if action.action_type == ActionType.PLAY_KNIGHT and action.payload.victim_id == 1)
    state = engine.apply_action(state, knight_action, seed=5)
    assert state.public_players[0].has_largest_army

    assert any(action.action_type == ActionType.DECLARE_VICTORY for action in engine.legal_actions(state))
    state = engine.apply_action(state, Action(ActionType.DECLARE_VICTORY))
    assert state.phase == TurnPhase.GAME_OVER
    assert state.winner_id == 0
