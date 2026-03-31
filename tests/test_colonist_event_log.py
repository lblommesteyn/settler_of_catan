"""Tests for Colonist event-log parsing and turn inference."""

from __future__ import annotations

import numpy as np

from catan.board.board import Resource
from catan.colonist_cv import context_ocr
from catan.colonist_cv.context_ocr import read_screen_context
from catan.colonist_cv.event_log import ColonistEventType, infer_turn_context_from_events, parse_dataset_state_change, parse_visible_log_lines
from catan.colonist_cv.schema import PlayerColor
from catan.full_solver.state import DevCardType, TurnPhase


COLOR_TO_PLAYER = {
    PlayerColor.RED: 0,
    PlayerColor.BLUE: 1,
    PlayerColor.ORANGE: 2,
    PlayerColor.WHITE: 3,
}


def test_parse_dataset_state_change_maps_core_events():
    state_change = {
        "currentState": {"completedTurns": 42},
        "gameLogState": {
            "0": {"text": {"type": 10, "playerColor": 2, "firstDice": 3, "secondDice": 4}},
            "1": {"text": {"type": 118, "playerColor": 2, "wantedCardEnums": [2], "offeredCardEnums": [1, 1]}},
            "2": {"text": {"type": 115, "playerColor": 2, "acceptingPlayerColor": 1, "givenCardEnums": [1, 1], "receivedCardEnums": [2]}},
            "3": {"text": {"type": 20, "playerColor": 2, "cardEnum": 11}},
            "4": {"text": {"type": 11, "playerColor": 2, "pieceEnum": 5}},
            "5": {"text": {"type": 16, "playerColorThief": 2, "playerColorVictim": 1, "cardBacks": [0]}},
            "6": {"text": {"type": 66, "playerColor": 2, "achievementEnum": 1}},
            "7": {"text": {"type": 1, "playerColor": 2}},
        },
    }

    events = parse_dataset_state_change(state_change, color_to_player=COLOR_TO_PLAYER)

    assert [event.event_type for event in events] == [
        ColonistEventType.ROLL,
        ColonistEventType.TRADE_OFFER,
        ColonistEventType.TRADE_ACCEPT,
        ColonistEventType.PLAY_DEV_CARD,
        ColonistEventType.MOVE_ROBBER,
        ColonistEventType.STEAL_CARD,
        ColonistEventType.CLAIM_ACHIEVEMENT,
        ColonistEventType.END_TURN,
    ]
    assert events[0].player_id == 0
    assert events[0].dice_total == 7
    assert events[1].given_resources == (Resource.BRICK, Resource.BRICK)
    assert events[1].received_resources == (Resource.SHEEP,)
    assert events[3].dev_card == DevCardType.KNIGHT
    assert events[6].achievement == "largest_army"
    assert all(event.completed_turns == 42 for event in events)


def test_infer_main_phase_from_visible_log_lines():
    events = parse_visible_log_lines(
        ["Blue rolled 8", "Blue built a road"],
        color_to_player=COLOR_TO_PLAYER,
    )
    context = infer_turn_context_from_events(events, color_to_player=COLOR_TO_PLAYER)

    assert [event.event_type for event in events] == [ColonistEventType.ROLL, ColonistEventType.BUILD]
    assert context.current_player == 1
    assert context.phase == TurnPhase.MAIN
    assert context.dice_rolled_this_turn is True
    assert context.last_roll == 8


def test_infer_pending_trade_from_visible_log_lines():
    events = parse_visible_log_lines(
        ["Blue rolled 6", "Blue wants to give 1 brick for 1 sheep"],
        color_to_player=COLOR_TO_PLAYER,
    )
    context = infer_turn_context_from_events(events, color_to_player=COLOR_TO_PLAYER)

    assert events[-1].event_type == ColonistEventType.TRADE_OFFER
    assert context.current_player == 1
    assert context.phase == TurnPhase.PENDING_TRADE
    assert context.dice_rolled_this_turn is True


def test_infer_setup_step_from_visible_log_lines():
    events = parse_visible_log_lines(
        [
            "Red placed a settlement",
            "Red placed a road",
            "Blue placed a settlement",
        ],
        color_to_player=COLOR_TO_PLAYER,
    )
    context = infer_turn_context_from_events(events, color_to_player=COLOR_TO_PLAYER)

    assert context.phase == TurnPhase.SETUP
    assert context.current_player == 1
    assert context.setup_step == 1
    assert context.dice_rolled_this_turn is False


def test_read_screen_context_falls_back_to_event_log(monkeypatch):
    monkeypatch.setattr(context_ocr, "_detect_prompt_text", lambda image: "")
    monkeypatch.setattr(context_ocr, "_detect_visible_log_lines", lambda image: ("Blue rolled 8", "Blue built a road"))

    detected = read_screen_context(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        my_color=PlayerColor.BLUE,
        color_to_player=COLOR_TO_PLAYER,
        player_id_hint=None,
    )

    assert detected.current_player == 1
    assert detected.phase == TurnPhase.MAIN
    assert detected.dice_rolled_this_turn is True
    assert detected.last_roll == 8
    assert detected.log_lines == ("Blue rolled 8", "Blue built a road")
    assert detected.recent_events[-1].event_type == ColonistEventType.BUILD


def test_read_screen_context_does_not_treat_accept_trade_prompt_as_local_turn(monkeypatch):
    monkeypatch.setattr(context_ocr, "_detect_prompt_text", lambda image: "accept trade")
    monkeypatch.setattr(context_ocr, "_detect_visible_log_lines", lambda image: ("Red wants to give 1 brick for 1 sheep",))

    detected = read_screen_context(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        my_color=PlayerColor.BLUE,
        color_to_player=COLOR_TO_PLAYER,
        player_id_hint=None,
    )

    assert detected.phase == TurnPhase.PENDING_TRADE
    assert detected.current_player == 0
    assert detected.dice_rolled_this_turn is True
