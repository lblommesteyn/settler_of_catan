"""OCR and CV helpers for reading live Colonist turn context from the screen."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..board.board import Resource
from ..full_solver.state import DevCardType, TurnPhase, empty_hand, playable_resources
from .detector import DefaultPlayerPalette, DefaultResourcePalette, DetectionError, PrototypeColorClassifier
from .ocr import OCRUnavailableError, read_text
from .schema import PlayerColor, PrivateObservation


PROMPT_REGION_RATIOS = (
    (0.38, 0.88, 0.72, 0.98),
    (0.42, 0.84, 0.76, 0.96),
    (0.34, 0.82, 0.80, 0.98),
)
HAND_REGION_RATIOS = (0.10, 0.90, 0.43, 0.995)
LOCAL_COLOR_REGION_RATIOS = (0.70, 0.89, 0.86, 0.995)


@dataclass(frozen=True)
class ScreenContextDetection:
    my_color: Optional[PlayerColor]
    current_player: Optional[int]
    phase: Optional[TurnPhase]
    private_pov: Optional[PrivateObservation]
    dice_rolled_this_turn: Optional[bool]
    prompt_text: str = ""


def _crop_ratio(image: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(round(width * region[0]))
    y0 = int(round(height * region[1]))
    x1 = int(round(width * region[2]))
    y1 = int(round(height * region[3]))
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return image[y0:y1, x0:x1]


def _text_variants(image: np.ndarray) -> tuple[np.ndarray, ...]:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim == 2:
        gray = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
    return (gray, enlarged, otsu, inv, adaptive)


def _ocr_joined_text(image: np.ndarray, *, allowlist: Optional[str] = None) -> str:
    fragments: list[str] = []
    for variant in _text_variants(image):
        try:
            results = read_text(variant, allowlist=allowlist, detail=0, paragraph=False)
        except OCRUnavailableError as exc:
            raise DetectionError(str(exc)) from exc
        for result in results:
            text = str(result).strip()
            if text:
                fragments.append(text)
    normalized = " ".join(fragments)
    return " ".join(normalized.lower().split())


def _extract_integer(image: np.ndarray) -> Optional[int]:
    text = _ocr_joined_text(image, allowlist="0123456789")
    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def _detect_prompt_text(image: np.ndarray) -> str:
    best = ""
    best_score = -1
    for region in PROMPT_REGION_RATIOS:
        crop = _crop_ratio(image, region)
        text = _ocr_joined_text(crop)
        score = sum(
            1
            for keyword in (
                "place settlement",
                "place road",
                "roll",
                "end turn",
                "discard",
                "robber",
                "trade",
            )
            if keyword in text
        )
        if score > best_score or (score == best_score and len(text) > len(best)):
            best = text
            best_score = score
    return best


def _phase_from_prompt(prompt_text: str) -> tuple[Optional[TurnPhase], Optional[bool]]:
    if not prompt_text:
        return None, None
    if "place settlement" in prompt_text or "place road" in prompt_text:
        return TurnPhase.SETUP, False
    if "discard" in prompt_text or "robber" in prompt_text:
        return TurnPhase.RESOLVE_SEVEN, True
    if "accept" in prompt_text and "trade" in prompt_text:
        return TurnPhase.PENDING_TRADE, True
    if "roll" in prompt_text:
        return TurnPhase.PRE_ROLL, False
    if "end turn" in prompt_text or "build" in prompt_text or "maritime" in prompt_text or "bank trade" in prompt_text:
        return TurnPhase.MAIN, True
    return None, None


def detect_local_player_color(image: np.ndarray) -> Optional[PlayerColor]:
    patch = _crop_ratio(image, LOCAL_COLOR_REGION_RATIOS)
    if patch.size == 0:
        return None
    rgb = np.asarray(patch, dtype=np.float32)
    votes: dict[PlayerColor, float] = {}
    for color in (PlayerColor.RED, PlayerColor.BLUE, PlayerColor.ORANGE, PlayerColor.GREEN):
        prototype = np.asarray(DefaultPlayerPalette[color.value], dtype=np.float32)
        dist = np.linalg.norm(rgb - prototype[None, None, :], axis=2)
        votes[color] = float(np.sum(np.clip(95.0 - dist, 0.0, None)))
    best_color, best_vote = max(votes.items(), key=lambda item: item[1])
    total = sum(votes.values())
    if best_vote <= 0.0 or best_vote < total * 0.28:
        return None
    return best_color


def _resource_mask(patch: np.ndarray, resource: Resource) -> np.ndarray:
    prototype = np.asarray(DefaultResourcePalette[resource.value], dtype=np.float32)
    rgb = np.asarray(patch, dtype=np.float32)
    dist = np.linalg.norm(rgb - prototype[None, None, :], axis=2)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    if resource == Resource.ORE:
        mask = (dist < 80.0) & (value > 80)
    elif resource == Resource.WHEAT:
        mask = (dist < 90.0) & (saturation > 35)
    else:
        mask = (dist < 95.0) & (saturation > 45)
    return mask.astype(np.uint8) * 255


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes)
    merged: list[list[int]] = []
    for x, y, w, h in boxes:
        x1 = x + w
        y1 = y + h
        if not merged:
            merged.append([x, y, x1, y1])
            continue
        last = merged[-1]
        overlap_x = min(last[2], x1) - max(last[0], x)
        overlap_y = min(last[3], y1) - max(last[1], y)
        if overlap_x >= -8 and overlap_y >= -12:
            last[0] = min(last[0], x)
            last[1] = min(last[1], y)
            last[2] = max(last[2], x1)
            last[3] = max(last[3], y1)
        else:
            merged.append([x, y, x1, y1])
    return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in merged]


def detect_hand_resources(image: np.ndarray) -> dict[Resource, int]:
    tray = _crop_ratio(image, HAND_REGION_RATIOS)
    counts = empty_hand()
    if tray.size == 0:
        return counts

    tray_height, tray_width = tray.shape[:2]
    kernel = np.ones((5, 5), dtype=np.uint8)
    classifier = PrototypeColorClassifier({resource.value: DefaultResourcePalette[resource.value] for resource in playable_resources()})
    used_boxes: list[tuple[int, int, int, int]] = []
    for resource in playable_resources():
        mask = _resource_mask(tray, resource)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        boxes: list[tuple[int, int, int, int]] = []
        for index in range(1, num_labels):
            x, y, width, height, area = stats[index]
            if area < 110:
                continue
            if width < 12 or height < 18:
                continue
            if width > tray_width * 0.18 or height > tray_height * 0.92:
                continue
            if area > tray_width * tray_height * 0.15:
                continue
            if y + height < tray_height * 0.15:
                continue
            boxes.append((int(x), int(y), int(width), int(height)))

        for x, y, width, height in _merge_boxes(boxes):
            box = (x, y, width, height)
            if any(
                abs(x - used_x) <= 10
                and abs(y - used_y) <= 10
                and abs(width - used_w) <= 12
                and abs(height - used_h) <= 12
                for used_x, used_y, used_w, used_h in used_boxes
            ):
                continue
            card = tray[y : y + height, x : x + width]
            if card.size == 0:
                continue
            mean_rgb = card.reshape(-1, 3).mean(axis=0)
            result = classifier.classify(mean_rgb)
            if result.label != resource.value or result.confidence < 0.20:
                continue
            digit_crop = card[0 : max(1, int(height * 0.52)), max(0, int(width * 0.45)) : width]
            count = _extract_integer(digit_crop)
            counts[resource] += count if count is not None and count > 0 else 1
            used_boxes.append(box)
    return counts


def read_screen_context(
    image: np.ndarray,
    *,
    my_color: Optional[PlayerColor],
    color_to_player: dict[PlayerColor, int],
    player_id_hint: Optional[int] = None,
) -> ScreenContextDetection:
    resolved_color = my_color or detect_local_player_color(image)
    prompt_text = _detect_prompt_text(image)
    phase, dice_rolled = _phase_from_prompt(prompt_text)

    player_id = player_id_hint
    if player_id is None and resolved_color is not None and resolved_color in color_to_player:
        player_id = color_to_player[resolved_color]

    current_player = None
    if phase is not None and player_id is not None:
        current_player = player_id

    private_pov = None
    if player_id is not None:
        private_pov = PrivateObservation(
            player_id=player_id,
            resources=detect_hand_resources(image),
            dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
            new_dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
            hidden_vp_cards=0,
        )

    return ScreenContextDetection(
        my_color=resolved_color,
        current_player=current_player,
        phase=phase,
        private_pov=private_pov,
        dice_rolled_this_turn=dice_rolled,
        prompt_text=prompt_text,
    )
