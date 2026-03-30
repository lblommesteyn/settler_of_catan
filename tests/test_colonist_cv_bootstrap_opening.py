"""Tests for the automatic Colonist board bootstrap and live opening flow."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from catan.board.board import CatanBoard, PortType, Resource
from catan.board.hex_grid import AXIAL_POSITIONS
from catan.colonist_cv import bootstrap as bootstrap_module
from catan.colonist_cv.bootstrap import RESOURCE_HSV_PROTOTYPES, AutoBoardBootstrap, auto_bootstrap_board
from catan.colonist_cv.geometry import BoardCalibration
from catan.colonist_cv.opening_live import PromptKind, ScreenPrompt, analyze_opening_screen, suggestion_target_text
from catan.colonist_cv.schema import PlayerColor, PublicStructures


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    hsv = np.uint8([[[int(round(hue)), int(round(saturation)), int(round(value))]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _synthetic_layout() -> tuple[dict[tuple[int, int], Resource], dict[tuple[int, int], int | None], tuple[int, int]]:
    desert_hex = (-2, 1)
    resources = {
        (0, 0): Resource.ORE,
        (1, 0): Resource.WHEAT,
        (0, 1): Resource.WHEAT,
        (-1, 1): Resource.WHEAT,
        (-1, 0): Resource.SHEEP,
        (0, -1): Resource.SHEEP,
        (1, -1): Resource.WOOD,
        (2, 0): Resource.WOOD,
        (1, 1): Resource.WOOD,
        (0, 2): Resource.SHEEP,
        (-1, 2): Resource.WHEAT,
        (-2, 2): Resource.ORE,
        (-2, 1): Resource.DESERT,
        (-2, 0): Resource.ORE,
        (-1, -1): Resource.WOOD,
        (0, -2): Resource.BRICK,
        (1, -2): Resource.SHEEP,
        (2, -2): Resource.BRICK,
        (2, -1): Resource.BRICK,
    }
    numbers = {
        (2, 0): 12,
        (2, -1): 11,
        (0, 0): 11,
        (0, 2): 10,
        (0, -1): 10,
        (1, 1): 9,
        (-1, 0): 9,
        (1, -2): 8,
        (-1, 2): 8,
        (1, 0): 6,
        (-2, 0): 6,
        (0, 1): 5,
        (0, -2): 5,
        (2, -2): 4,
        (-1, 1): 4,
        (1, -1): 3,
        (-2, 2): 3,
        (-1, -1): 2,
        desert_hex: None,
    }
    return resources, numbers, desert_hex


def _synthetic_hex_centers() -> dict[tuple[int, int], tuple[float, float]]:
    tx = 762.0
    ty = 756.0
    dx = 145.0
    dy = 125.0
    return {
        hex_coord: (tx + dx * (hex_coord[0] + 0.5 * hex_coord[1]), ty + dy * hex_coord[1])
        for hex_coord in AXIAL_POSITIONS
    }


def _draw_token_square(image: np.ndarray, center: tuple[float, float]) -> None:
    cx, cy = (int(round(center[0])), int(round(center[1])))
    cv2.rectangle(image, (cx - 27, cy - 27), (cx + 27, cy + 27), (255, 255, 255), -1)


def _build_synthetic_bootstrap_image() -> np.ndarray:
    resources, numbers, desert_hex = _synthetic_layout()
    centers = _synthetic_hex_centers()
    image = np.zeros((1400, 1800, 3), dtype=np.uint8)
    image[:] = (58, 116, 179)
    for hex_coord, center in centers.items():
        resource = resources[hex_coord]
        color = _hsv_to_rgb(*RESOURCE_HSV_PROTOTYPES[resource])
        cv2.circle(image, (int(round(center[0])), int(round(center[1]))), 92, color, -1)
    for hex_coord, center in centers.items():
        if hex_coord == desert_hex:
            continue
        _draw_token_square(image, center)
    return image


def test_auto_bootstrap_board_recovers_synthetic_layout(monkeypatch):
    image = _build_synthetic_bootstrap_image()
    resources, numbers, desert_hex = _synthetic_layout()
    centers = _synthetic_hex_centers()

    def fake_number_votes(_image, center, _radius):
        nearest = min(centers, key=lambda hex_coord: np.linalg.norm(np.asarray(center) - np.asarray(centers[hex_coord])))
        number = numbers[nearest]
        return Counter() if number is None else Counter({int(number): 5})

    monkeypatch.setattr(bootstrap_module, "_number_votes", fake_number_votes)

    bootstrap = auto_bootstrap_board(image)

    assert bootstrap.desert_hex == desert_hex
    assert bootstrap.numbers_by_hex[desert_hex] is None
    mismatches = [
        hex_coord
        for hex_coord in AXIAL_POSITIONS
        if bootstrap.resources_by_hex[hex_coord] != resources[hex_coord]
    ]
    assert len(mismatches) <= 1
    for hex_coord, number in numbers.items():
        assert bootstrap.numbers_by_hex[hex_coord] == number


def test_analyze_opening_screen_uses_setup_state_and_returns_suggestions(monkeypatch):
    board = CatanBoard.from_tiles(
        AXIAL_POSITIONS,
        [Resource.WOOD if hex_coord != (-2, 1) else Resource.DESERT for hex_coord in AXIAL_POSITIONS],
        [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
        [PortType.GENERIC] * 9,
    )
    calibration = BoardCalibration.from_matrices(
        board,
        canonical_to_screen=np.array(
            [
                [83.0, 0.0, 760.0],
                [0.0, 83.0, 760.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    bootstrap = AutoBoardBootstrap(
        board=board,
        calibration=calibration,
        desert_hex=(-2, 1),
        token_candidates=tuple(),
        token_centers={},
        numbers_by_hex={hex_coord: None for hex_coord in AXIAL_POSITIONS},
        resources_by_hex={hex_coord: (Resource.DESERT if hex_coord == (-2, 1) else Resource.WOOD) for hex_coord in AXIAL_POSITIONS},
    )
    image = np.zeros((1200, 1600, 3), dtype=np.uint8)
    image[:] = (45, 92, 140)
    orange_v = (-2.598076, 0.5)
    green_v = (-0.866025, 2.5)
    blue_v = (1.732051, 1.0)

    monkeypatch.setattr("catan.colonist_cv.opening_live.auto_bootstrap_board", lambda _image: bootstrap)
    monkeypatch.setattr(
        "catan.colonist_cv.opening_live._detect_prompt",
        lambda _image: ScreenPrompt(text="Place Settlement", kind=PromptKind.PLACE_SETTLEMENT, bbox=(100, 100, 200, 130), my_color=None),
    )
    monkeypatch.setattr("catan.colonist_cv.opening_live._infer_prompt_color", lambda _image, _prompt: PlayerColor.RED)
    monkeypatch.setattr(
        "catan.colonist_cv.opening_live._detect_public_players",
        lambda _image, _bootstrap: (
            PublicStructures(player_id=0, color=PlayerColor.RED),
            PublicStructures(player_id=1, color=PlayerColor.BLUE, settlements=frozenset({blue_v})),
            PublicStructures(player_id=2, color=PlayerColor.ORANGE, settlements=frozenset({orange_v})),
            PublicStructures(player_id=3, color=PlayerColor.GREEN, settlements=frozenset({green_v})),
        ),
    )

    analysis = analyze_opening_screen(image, my_color=PlayerColor.RED, top_k=3)

    assert analysis.seat == 3
    assert analysis.my_color == PlayerColor.RED
    assert len(analysis.suggestions) == 3
    assert all(suggestion.kind == PromptKind.PLACE_SETTLEMENT for suggestion in analysis.suggestions)
    assert analysis.suggestions[0].summary.startswith("First setup settlement.")
    assert analysis.suggestions[0].plan is not None
    assert analysis.suggestions[0].plan.startswith("Lean: ")

    target_text = suggestion_target_text(analysis, analysis.suggestions[0])
    assert target_text.startswith("Click ")
    assert " @ (" in target_text
