"""Tests for OCR-driven turn context extraction from Colonist screenshots."""

from __future__ import annotations

import cv2
import numpy as np

from catan.board.board import Resource
from catan.colonist_cv.context_ocr import (
    HAND_REGION_RATIOS,
    LOCAL_COLOR_REGION_RATIOS,
    detect_hand_resources,
    detect_local_player_color,
    read_screen_context,
)
from catan.colonist_cv.detector import DefaultPlayerPalette, DefaultResourcePalette
from catan.colonist_cv.schema import PlayerColor
from catan.full_solver.state import TurnPhase


def _resource_box(image: np.ndarray, slot: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    x0 = int(width * HAND_REGION_RATIOS[0])
    y0 = int(height * HAND_REGION_RATIOS[1])
    x1 = int(width * HAND_REGION_RATIOS[2])
    y1 = int(height * HAND_REGION_RATIOS[3])
    tray_width = x1 - x0
    tray_height = y1 - y0
    box_w = int(tray_width * 0.11)
    box_h = int(tray_height * 0.72)
    gap = int(tray_width * 0.03)
    left = x0 + 18 + slot * (box_w + gap)
    top = y0 + int(tray_height * 0.12)
    cv2.rectangle(image, (left, top), (left + box_w, top + box_h), color, -1)


def test_detect_local_player_color_from_synthetic_panel():
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[:] = (50, 93, 140)
    x0 = int(image.shape[1] * LOCAL_COLOR_REGION_RATIOS[0])
    y0 = int(image.shape[0] * LOCAL_COLOR_REGION_RATIOS[1])
    x1 = int(image.shape[1] * LOCAL_COLOR_REGION_RATIOS[2])
    y1 = int(image.shape[0] * LOCAL_COLOR_REGION_RATIOS[3])
    cv2.circle(
        image,
        ((x0 + x1) // 2, (y0 + y1) // 2),
        min(x1 - x0, y1 - y0) // 3,
        DefaultPlayerPalette[PlayerColor.RED.value],
        -1,
    )

    assert detect_local_player_color(image) == PlayerColor.RED


def test_detect_hand_resources_from_synthetic_tray(monkeypatch):
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[:] = (50, 93, 140)
    x0 = int(image.shape[1] * HAND_REGION_RATIOS[0])
    y0 = int(image.shape[0] * HAND_REGION_RATIOS[1])
    x1 = int(image.shape[1] * HAND_REGION_RATIOS[2])
    y1 = int(image.shape[0] * HAND_REGION_RATIOS[3])
    cv2.rectangle(image, (x0, y0), (x1, y1), (232, 224, 206), -1)

    _resource_box(image, 0, DefaultResourcePalette[Resource.WOOD.value])
    _resource_box(image, 1, DefaultResourcePalette[Resource.WHEAT.value])
    _resource_box(image, 2, DefaultResourcePalette[Resource.ORE.value])

    digit_values = iter([2, 3, 1])
    monkeypatch.setattr("catan.colonist_cv.context_ocr._extract_integer", lambda _crop: next(digit_values))

    counts = detect_hand_resources(image)

    assert counts[Resource.WOOD] == 2
    assert counts[Resource.WHEAT] == 3
    assert counts[Resource.ORE] == 1
    assert counts[Resource.BRICK] == 0


def test_read_screen_context_maps_prompt_and_hand(monkeypatch):
    image = np.zeros((900, 1600, 3), dtype=np.uint8)
    image[:] = (50, 93, 140)
    color_to_player = {
        PlayerColor.RED: 0,
        PlayerColor.BLUE: 1,
        PlayerColor.ORANGE: 2,
        PlayerColor.GREEN: 3,
    }

    monkeypatch.setattr("catan.colonist_cv.context_ocr._detect_prompt_text", lambda _image: "roll dice")
    monkeypatch.setattr(
        "catan.colonist_cv.context_ocr.detect_hand_resources",
        lambda _image: {
            Resource.WOOD: 2,
            Resource.BRICK: 1,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        },
    )

    detected = read_screen_context(image, my_color=PlayerColor.RED, color_to_player=color_to_player)

    assert detected.my_color == PlayerColor.RED
    assert detected.current_player == 0
    assert detected.phase == TurnPhase.PRE_ROLL
    assert detected.dice_rolled_this_turn is False
    assert detected.private_pov is not None
    assert detected.private_pov.resources[Resource.WOOD] == 2
    assert detected.private_pov.resources[Resource.BRICK] == 1
