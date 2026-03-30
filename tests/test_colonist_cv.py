"""Tests for the Colonist computer-vision integration layer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from catan.board.board import CatanBoard, Resource
from catan.colonist_cv.advisor import HeuristicActionAdvisor
from catan.colonist_cv.detector import ColonistVisionDetector, DefaultPlayerPalette
from catan.colonist_cv.geometry import BoardCalibration
from catan.colonist_cv.runtime import (
    ScreenRegion,
    apply_context_overrides,
    fingerprint_observation,
    format_strategy_lines,
    load_live_context,
)
from catan.colonist_cv.schema import PlayerColor, PrivateObservation, PublicStructures, VisionFrameObservation
from catan.colonist_cv.tracker import build_state_from_observation
from catan.full_solver.actions import ActionType
from catan.full_solver.engine import ExactRulesEngine
from catan.full_solver.rules import refresh_public_state
from catan.full_solver.state import ExactGameState, PrivatePlayerState, PublicPlayerState, TurnPhase, empty_hand, make_bank_resources, make_dev_deck


def _draw_disk(image: np.ndarray, center: tuple[float, float], radius: float, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    cx, cy = center
    ys, xs = np.mgrid[0:height, 0:width]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    image[mask] = color


def _draw_segment(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    half_width: float,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    sx, sy = start
    ex, ey = end
    direction = np.array([ex - sx, ey - sy], dtype=float)
    length_sq = float(np.dot(direction, direction))
    rel = np.stack([xs - sx, ys - sy], axis=-1)
    t = np.clip((rel[..., 0] * direction[0] + rel[..., 1] * direction[1]) / max(length_sq, 1e-9), 0.0, 1.0)
    closest_x = sx + t * direction[0]
    closest_y = sy + t * direction[1]
    mask = (xs - closest_x) ** 2 + (ys - closest_y) ** 2 <= half_width ** 2
    image[mask] = color


def _sample_board() -> CatanBoard:
    return CatanBoard.random(seed=7)


def _sample_calibration(board: CatanBoard) -> BoardCalibration:
    anchors = {
        (-2, 0): (120.0, 300.0),
        (2, 0): (660.0, 260.0),
        (0, 2): (420.0, 600.0),
        (0, -2): (360.0, 80.0),
    }
    return BoardCalibration.from_hex_anchors(board, anchors)


def test_board_calibration_round_trip():
    board = _sample_board()
    calibration = _sample_calibration(board)

    vertex = board.all_vertices()[10]
    projected = calibration.project_vertex(vertex)
    recovered = calibration.unproject_point(projected)

    assert np.allclose(recovered, vertex, atol=1e-5)
    assert calibration.nearest_vertex(projected) == vertex

    hex_coord = list(board.tiles)[5]
    assert calibration.nearest_hex(calibration.project_hex_center(hex_coord)) == hex_coord


def test_detector_reads_synthetic_public_structures_and_robber():
    board = _sample_board()
    calibration = _sample_calibration(board)
    detector = ColonistVisionDetector()
    color_to_player = {
        PlayerColor.RED: 0,
        PlayerColor.BLUE: 1,
        PlayerColor.ORANGE: 2,
        PlayerColor.WHITE: 3,
    }

    image = np.zeros((720, 840, 3), dtype=np.uint8)
    image[:] = DefaultPlayerPalette["background"]
    scale = calibration.infer_scale()

    settlement_vertex = board.all_vertices()[3]
    city_vertex = board.all_vertices()[18]
    road_edge = next(
        edge for edge in board.graph.edges
        if settlement_vertex not in edge and city_vertex not in edge
    )
    robber_hex = list(board.tiles)[8]

    _draw_disk(image, calibration.project_vertex(settlement_vertex), scale * 0.16, DefaultPlayerPalette[PlayerColor.RED.value])
    _draw_disk(image, calibration.project_vertex(city_vertex), scale * 0.23, DefaultPlayerPalette[PlayerColor.BLUE.value])
    v1, v2 = tuple(road_edge)
    _draw_segment(
        image,
        calibration.project_vertex(v1),
        calibration.project_vertex(v2),
        scale * 0.11,
        DefaultPlayerPalette[PlayerColor.ORANGE.value],
    )
    _draw_disk(image, calibration.project_hex_center(robber_hex), scale * 0.12, (0, 0, 0))

    public_players = detector.detect_public_structures(image, calibration, color_to_player)
    robber_detected = detector.detect_robber_hex(image, calibration)

    assert settlement_vertex in public_players[0].settlements
    assert city_vertex in public_players[1].cities
    assert road_edge in public_players[2].roads
    assert robber_detected == robber_hex


def test_tracker_preserves_hidden_state_and_overlays_public_frame():
    board = _sample_board()
    previous = ExactGameState(
        board=board,
        public_players=tuple(PublicPlayerState(player_id=player_id) for player_id in range(4)),
        private_players=(
            PrivatePlayerState(player_id=0, resources={**empty_hand(), Resource.WOOD: 2}, dev_cards_in_hand={}, new_dev_cards_in_hand={}, hidden_vp_cards=0).canonical(),
            PrivatePlayerState(player_id=1, resources={**empty_hand(), Resource.ORE: 3}, dev_cards_in_hand={}, new_dev_cards_in_hand={}, hidden_vp_cards=1).canonical(),
            PrivatePlayerState(player_id=2, resources=empty_hand(), dev_cards_in_hand={}, new_dev_cards_in_hand={}, hidden_vp_cards=0).canonical(),
            PrivatePlayerState(player_id=3, resources=empty_hand(), dev_cards_in_hand={}, new_dev_cards_in_hand={}, hidden_vp_cards=0).canonical(),
        ),
        current_player=1,
        phase=TurnPhase.MAIN,
        robber_hex=board.robber_start,
        bank_resources=make_bank_resources(),
        dev_deck=make_dev_deck(seed=11),
    )

    v0 = board.all_vertices()[0]
    observation = VisionFrameObservation(
        board=board,
        robber_hex=list(board.tiles)[1],
        public_players=(
            PublicStructures(player_id=0, settlements=frozenset({v0})),
            PublicStructures(player_id=1),
            PublicStructures(player_id=2),
            PublicStructures(player_id=3),
        ),
        current_player=0,
        phase=TurnPhase.PRE_ROLL,
        private_pov=PrivateObservation(
            player_id=0,
            resources={**empty_hand(), Resource.WOOD: 1, Resource.BRICK: 1},
        ),
    )

    state = build_state_from_observation(observation, previous_state=previous)

    assert state.current_player == 0
    assert state.phase == TurnPhase.PRE_ROLL
    assert state.public_players[0].settlements == frozenset({v0})
    assert state.private_players[0].resources[Resource.WOOD] == 1
    assert state.private_players[1].resources[Resource.ORE] == 3
    assert state.private_players[1].hidden_vp_cards == 1


def test_advisor_prefers_building_settlement_over_ending_turn():
    board = _sample_board()
    engine = ExactRulesEngine()
    start = board.legal_starting_vertices()[0]
    middle = board.graph.vertex_neighbors[start][0]
    target = max(
        (
            vertex for vertex in board.graph.vertex_neighbors[middle]
            if vertex != start and start not in board.graph.vertex_neighbors[vertex]
        ),
        key=board.pip_count,
    )

    public_players = [
        PublicPlayerState(
            player_id=0,
            settlements=frozenset({start}),
            roads=frozenset({frozenset({start, middle}), frozenset({middle, target})}),
            settlements_left=4,
            roads_left=13,
        ),
        PublicPlayerState(player_id=1),
        PublicPlayerState(player_id=2),
        PublicPlayerState(player_id=3),
    ]
    private_players = [
        PrivatePlayerState(
            player_id=0,
            resources={
                Resource.WOOD: 1,
                Resource.BRICK: 1,
                Resource.SHEEP: 1,
                Resource.WHEAT: 1,
                Resource.ORE: 0,
            },
            dev_cards_in_hand={},
            new_dev_cards_in_hand={},
            hidden_vp_cards=0,
        ).canonical(),
        PrivatePlayerState(player_id=1).canonical(),
        PrivatePlayerState(player_id=2).canonical(),
        PrivatePlayerState(player_id=3).canonical(),
    ]

    state = refresh_public_state(
        ExactGameState(
            board=board,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            current_player=0,
            phase=TurnPhase.MAIN,
            robber_hex=board.robber_start,
            bank_resources=make_bank_resources(),
            dev_deck=make_dev_deck(seed=2),
            dice_rolled_this_turn=True,
        )
    )

    suggestions = HeuristicActionAdvisor(engine).suggest(state, top_k=3)

    assert suggestions
    assert suggestions[0].action.action_type == ActionType.BUILD_SETTLEMENT
    assert suggestions[0].action.payload == target

    plan = HeuristicActionAdvisor(engine).strategy_plan(state)
    assert "Tempo" in plan.lean or "Expansion" in plan.lean
    assert "settlement" in plan.buy_priority.lower()
    assert "Goal:" in format_strategy_lines(plan)[2]


def test_live_context_loader_and_override_merge(tmp_path):
    context_path = tmp_path / "context.json"
    context_path.write_text(
        """
        {
          "current_player": 2,
          "phase": "main",
          "private_pov": {
            "player_id": 0,
            "resources": {"wood": 2, "brick": 1},
            "dev_cards_in_hand": {"knight": 1},
            "new_dev_cards_in_hand": {"monopoly": 1},
            "hidden_vp_cards": 1
          },
          "public_overrides": [
            {"player_id": 1, "visible_vp": 5, "played_knights": 2, "dev_cards_bought": 3}
          ],
          "last_roll": 8,
          "dice_rolled_this_turn": true
        }
        """
    )
    context = load_live_context(context_path)
    assert context.current_player == 2
    assert context.phase == TurnPhase.MAIN
    assert context.private_pov is not None
    assert context.private_pov.resources[Resource.WOOD] == 2
    assert context.private_pov.dev_cards_in_hand

    board = _sample_board()
    observation = VisionFrameObservation(
        board=board,
        robber_hex=board.robber_start,
        public_players=(
            PublicStructures(player_id=0),
            PublicStructures(player_id=1),
            PublicStructures(player_id=2),
            PublicStructures(player_id=3),
        ),
        current_player=0,
        phase=TurnPhase.PRE_ROLL,
    )
    merged = apply_context_overrides(observation, context)
    assert merged.current_player == 2
    assert merged.phase == TurnPhase.MAIN
    assert merged.public_players[1].visible_vp == 5
    assert merged.public_players[1].played_knights == 2


def test_screen_region_parse_and_fingerprint_changes_with_private_hand():
    region = ScreenRegion.parse("10,20,110,220")
    assert region.as_bbox() == (10, 20, 110, 220)

    board = _sample_board()
    observation = VisionFrameObservation(
        board=board,
        robber_hex=board.robber_start,
        public_players=(
            PublicStructures(player_id=0),
            PublicStructures(player_id=1),
            PublicStructures(player_id=2),
            PublicStructures(player_id=3),
        ),
        current_player=0,
        phase=TurnPhase.MAIN,
    )
    before = fingerprint_observation(observation)
    after = fingerprint_observation(
        replace(
            observation,
            private_pov=PrivateObservation(
                player_id=0,
                resources={**empty_hand(), Resource.WOOD: 1},
            ),
        )
    )
    assert before != after
