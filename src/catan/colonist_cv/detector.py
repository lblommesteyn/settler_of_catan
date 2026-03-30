"""Sample-based screenshot detector for Colonist board state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..board.board import CatanBoard, Resource
from ..full_solver.state import TurnPhase
from .geometry import BoardCalibration
from .schema import PlayerColor, PrivateObservation, PublicStructures, VisionFrameObservation


class DetectionError(RuntimeError):
    """Raised when a screenshot cannot be mapped reliably into game state."""


def _clamp_bounds(height: int, width: int, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    return (
        max(0, x0),
        max(0, y0),
        min(width, x1),
        min(height, y1),
    )


def _sample_disk(image: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    height, width = image.shape[:2]
    cx, cy = center
    x0, y0, x1, y1 = _clamp_bounds(
        height,
        width,
        int(cx - radius - 1),
        int(cy - radius - 1),
        int(cx + radius + 2),
        int(cy + radius + 2),
    )
    if x0 >= x1 or y0 >= y1:
        return np.empty((0, image.shape[2]), dtype=float)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    patch = image[y0:y1, x0:x1]
    return patch[mask]


def _sample_segment(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    half_width: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(min(start[0], end[0]) - half_width - 1)
    x1 = int(max(start[0], end[0]) + half_width + 2)
    y0 = int(min(start[1], end[1]) - half_width - 1)
    y1 = int(max(start[1], end[1]) + half_width + 2)
    x0, y0, x1, y1 = _clamp_bounds(height, width, x0, y0, x1, y1)
    if x0 >= x1 or y0 >= y1:
        return np.empty((0, image.shape[2]), dtype=float)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    sx, sy = start
    ex, ey = end
    direction = np.array([ex - sx, ey - sy], dtype=float)
    length_sq = float(np.dot(direction, direction))
    if length_sq < 1e-12:
        return _sample_disk(image, start, half_width)
    rel = np.stack([xs - sx, ys - sy], axis=-1)
    t = np.clip((rel[..., 0] * direction[0] + rel[..., 1] * direction[1]) / length_sq, 0.0, 1.0)
    closest_x = sx + t * direction[0]
    closest_y = sy + t * direction[1]
    mask = ((xs - closest_x) ** 2 + (ys - closest_y) ** 2 <= half_width ** 2) & (t >= 0.2) & (t <= 0.8)
    patch = image[y0:y1, x0:x1]
    return patch[mask]


def _mean_rgb(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(samples, dtype=float).mean(axis=0)


def _foreground_mean(samples: np.ndarray, background_rgb: np.ndarray, min_distance: float = 24.0) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(3, dtype=float)
    samples = np.asarray(samples, dtype=float)
    distances = np.linalg.norm(samples - np.asarray(background_rgb, dtype=float), axis=1)
    foreground = samples[distances >= min_distance]
    if foreground.size == 0:
        return _mean_rgb(samples)
    return foreground.mean(axis=0)


@dataclass(frozen=True)
class ClassifiedColor:
    label: Optional[str]
    confidence: float


@dataclass(frozen=True)
class PrototypeColorClassifier:
    """Simple RGB prototype classifier for synthetic or tuned Colonist assets."""

    prototypes: dict[str, tuple[int, int, int]]
    background_label: Optional[str] = None

    def classify(self, rgb: np.ndarray) -> ClassifiedColor:
        if not self.prototypes:
            return ClassifiedColor(label=None, confidence=0.0)
        rgb = np.asarray(rgb, dtype=float)
        best_label = None
        best_distance = None
        second_distance = None
        for label, proto in self.prototypes.items():
            distance = float(np.linalg.norm(rgb - np.asarray(proto, dtype=float)))
            if best_distance is None or distance < best_distance:
                second_distance = best_distance
                best_distance = distance
                best_label = label
            elif second_distance is None or distance < second_distance:
                second_distance = distance
        if best_distance is None:
            return ClassifiedColor(label=None, confidence=0.0)
        if second_distance is None:
            confidence = 1.0
        else:
            confidence = max(0.0, min(1.0, 1.0 - (best_distance / max(second_distance, 1e-6))))
        if best_label == self.background_label:
            return ClassifiedColor(label=None, confidence=confidence)
        return ClassifiedColor(label=best_label, confidence=confidence)


DefaultPlayerPalette = {
    PlayerColor.RED.value: (198, 62, 64),
    PlayerColor.BLUE.value: (67, 115, 239),
    PlayerColor.ORANGE.value: (211, 164, 85),
    PlayerColor.GREEN.value: (94, 214, 60),
    PlayerColor.WHITE.value: (230, 232, 236),
    "background": (32, 32, 40),
}

DefaultResourcePalette = {
    Resource.WOOD.value: (74, 120, 56),
    Resource.BRICK.value: (178, 82, 56),
    Resource.SHEEP.value: (146, 192, 102),
    Resource.WHEAT.value: (212, 186, 77),
    Resource.ORE.value: (121, 127, 135),
    Resource.DESERT.value: (204, 181, 138),
}


@dataclass
class ColonistVisionDetector:
    """
    Sample-based CV detector.

    This is designed to be tuned with real screenshots. It already supports
    calibration-based geometry and deterministic site sampling, while leaving
    number and port readers pluggable.
    """

    player_classifier: PrototypeColorClassifier = field(
        default_factory=lambda: PrototypeColorClassifier(DefaultPlayerPalette, background_label="background")
    )
    resource_classifier: PrototypeColorClassifier = field(
        default_factory=lambda: PrototypeColorClassifier(DefaultResourcePalette)
    )
    min_piece_confidence: float = 0.40
    min_resource_confidence: float = 0.35

    def detect_public_structures(
        self,
        image: np.ndarray,
        calibration: BoardCalibration,
        color_to_player: dict[PlayerColor, int],
    ) -> tuple[PublicStructures, ...]:
        scale = calibration.infer_scale()
        vertex_radius = scale * 0.25
        edge_half_width = scale * 0.12
        edge_map = calibration.board.graph.edges

        settlements: dict[int, set[tuple[float, float]]] = {player_id: set() for player_id in color_to_player.values()}
        cities: dict[int, set[tuple[float, float]]] = {player_id: set() for player_id in color_to_player.values()}
        roads: dict[int, set[frozenset]] = {player_id: set() for player_id in color_to_player.values()}
        background_rgb = np.asarray(
            self.player_classifier.prototypes.get(self.player_classifier.background_label or "", (0, 0, 0)),
            dtype=float,
        )

        for vertex, point in calibration.vertices().items():
            samples = _sample_disk(image, point, vertex_radius)
            rgb = _foreground_mean(samples, background_rgb)
            result = self.player_classifier.classify(rgb)
            if result.label is None or result.confidence < self.min_piece_confidence:
                continue
            player_color = PlayerColor.from_value(result.label)
            player_id = color_to_player[player_color]
            occupancy = self._occupancy_fraction(image, point, vertex_radius, rgb)
            if occupancy >= 0.58:
                cities[player_id].add(vertex)
            else:
                settlements[player_id].add(vertex)

        for edge, (v1, v2) in edge_map.items():
            p1 = calibration.project_vertex(v1)
            p2 = calibration.project_vertex(v2)
            samples = _sample_segment(image, p1, p2, edge_half_width)
            rgb = _foreground_mean(samples, background_rgb)
            result = self.player_classifier.classify(rgb)
            if result.label is None or result.confidence < self.min_piece_confidence:
                continue
            player_color = PlayerColor.from_value(result.label)
            player_id = color_to_player[player_color]
            roads[player_id].add(edge)

        public_players = []
        for player_id in sorted(color_to_player.values()):
            color = next(color for color, mapped_id in color_to_player.items() if mapped_id == player_id)
            public_players.append(
                PublicStructures(
                    player_id=player_id,
                    color=color,
                    settlements=frozenset(settlements[player_id]),
                    cities=frozenset(cities[player_id]),
                    roads=frozenset(roads[player_id]),
                )
            )
        return tuple(public_players)

    def detect_tile_resources(
        self,
        image: np.ndarray,
        calibration: BoardCalibration,
    ) -> dict[tuple[int, int], Resource]:
        scale = calibration.infer_scale()
        radius = scale * 0.45
        resources = {}
        for hex_coord, center in calibration.hex_centers().items():
            rgb = _mean_rgb(_sample_disk(image, center, radius))
            result = self.resource_classifier.classify(rgb)
            if result.label is None or result.confidence < self.min_resource_confidence:
                raise DetectionError(f"Could not classify resource at hex {hex_coord}.")
            resources[hex_coord] = Resource(result.label)
        return resources

    def detect_robber_hex(
        self,
        image: np.ndarray,
        calibration: BoardCalibration,
    ) -> tuple[int, int]:
        scale = calibration.infer_scale()
        radius = scale * 0.18
        best_hex = None
        best_darkness = None
        for hex_coord, center in calibration.hex_centers().items():
            samples = _sample_disk(image, center, radius)
            if samples.size == 0:
                continue
            mean_luma = float(np.asarray(samples, dtype=float).mean())
            if best_darkness is None or mean_luma < best_darkness:
                best_darkness = mean_luma
                best_hex = hex_coord
        if best_hex is None:
            raise DetectionError("Could not detect robber position.")
        return best_hex

    def detect_frame(
        self,
        image: np.ndarray,
        board: CatanBoard,
        calibration: BoardCalibration,
        color_to_player: dict[PlayerColor, int],
        current_player: int,
        phase: TurnPhase,
        private_pov: Optional[PrivateObservation] = None,
    ) -> VisionFrameObservation:
        public_players = self.detect_public_structures(image, calibration, color_to_player)
        robber_hex = self.detect_robber_hex(image, calibration)
        return VisionFrameObservation(
            board=board,
            robber_hex=robber_hex,
            public_players=public_players,
            current_player=current_player,
            phase=phase,
            private_pov=private_pov,
        )

    def _occupancy_fraction(
        self,
        image: np.ndarray,
        point: tuple[float, float],
        radius: float,
        reference_rgb: np.ndarray,
    ) -> float:
        samples = _sample_disk(image, point, radius)
        if samples.size == 0:
            return 0.0
        distances = np.linalg.norm(np.asarray(samples, dtype=float) - np.asarray(reference_rgb, dtype=float), axis=1)
        return float(np.mean(distances < 55.0))
