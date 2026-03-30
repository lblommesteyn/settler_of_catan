"""Board calibration and geometric mapping utilities for screenshot-space inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..board.board import CatanBoard
from ..board.hex_grid import axial_to_cartesian


Point = tuple[float, float]


def _to_homogeneous(point: Point) -> np.ndarray:
    return np.array([point[0], point[1], 1.0], dtype=float)


def apply_homography(matrix: np.ndarray, point: Point) -> Point:
    projected = matrix @ _to_homogeneous(point)
    if abs(projected[2]) < 1e-12:
        raise ValueError("Degenerate homography projected point at infinity.")
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def estimate_homography(source_points: list[Point], target_points: list[Point]) -> np.ndarray:
    """
    Estimate a projective transform from at least four point correspondences.

    Uses the standard DLT construction solved by SVD.
    """

    if len(source_points) != len(target_points):
        raise ValueError("Source and target point lists must be the same length.")
    if len(source_points) < 4:
        raise ValueError("At least four point correspondences are required.")

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source_points, target_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    matrix = np.asarray(rows, dtype=float)
    _, _, vh = np.linalg.svd(matrix)
    homography = vh[-1].reshape(3, 3)
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("Estimated homography is degenerate.")
    return homography / homography[2, 2]


@dataclass(frozen=True)
class BoardCalibration:
    """Projective mapping between canonical board coordinates and screenshot pixels."""

    board: CatanBoard
    canonical_to_screen: np.ndarray
    screen_to_canonical: np.ndarray

    @classmethod
    def from_hex_anchors(
        cls,
        board: CatanBoard,
        anchors: dict[tuple[int, int], Point],
    ) -> "BoardCalibration":
        if len(anchors) < 4:
            raise ValueError("Need at least four hex-center anchors to calibrate the board.")
        source = [axial_to_cartesian(q, r) for q, r in anchors]
        target = [anchors[hex_coord] for hex_coord in anchors]
        forward = estimate_homography(source, target)
        inverse = np.linalg.inv(forward)
        return cls(board=board, canonical_to_screen=forward, screen_to_canonical=inverse)

    @classmethod
    def from_serialized(cls, board: CatanBoard, data: dict) -> "BoardCalibration":
        forward = np.asarray(data["canonical_to_screen"], dtype=float)
        inverse = np.asarray(data["screen_to_canonical"], dtype=float)
        return cls(board=board, canonical_to_screen=forward, screen_to_canonical=inverse)

    @classmethod
    def from_matrices(
        cls,
        board: CatanBoard,
        canonical_to_screen: np.ndarray,
        screen_to_canonical: np.ndarray | None = None,
    ) -> "BoardCalibration":
        forward = np.asarray(canonical_to_screen, dtype=float)
        inverse = (
            np.asarray(screen_to_canonical, dtype=float)
            if screen_to_canonical is not None
            else np.linalg.inv(forward)
        )
        return cls(board=board, canonical_to_screen=forward, screen_to_canonical=inverse)

    def to_serialized(self) -> dict:
        return {
            "canonical_to_screen": self.canonical_to_screen.tolist(),
            "screen_to_canonical": self.screen_to_canonical.tolist(),
        }

    def project_point(self, point: Point) -> Point:
        return apply_homography(self.canonical_to_screen, point)

    def unproject_point(self, point: Point) -> Point:
        return apply_homography(self.screen_to_canonical, point)

    def project_hex_center(self, hex_coord: tuple[int, int]) -> Point:
        return self.project_point(axial_to_cartesian(*hex_coord))

    def project_vertex(self, vertex: tuple[float, float]) -> Point:
        return self.project_point(vertex)

    def project_edge_midpoint(self, edge: frozenset) -> Point:
        v1, v2 = tuple(edge)
        midpoint = ((v1[0] + v2[0]) / 2.0, (v1[1] + v2[1]) / 2.0)
        return self.project_point(midpoint)

    def hex_centers(self) -> dict[tuple[int, int], Point]:
        return {hex_coord: self.project_hex_center(hex_coord) for hex_coord in self.board.tiles}

    def vertices(self) -> dict[tuple[float, float], Point]:
        return {vertex: self.project_vertex(vertex) for vertex in self.board.all_vertices()}

    def edge_midpoints(self) -> dict[frozenset, Point]:
        return {edge: self.project_edge_midpoint(edge) for edge in self.board.graph.edges}

    def infer_scale(self) -> float:
        lengths = []
        for edge in self.board.graph.edges:
            v1, v2 = tuple(edge)
            p1 = np.asarray(self.project_vertex(v1), dtype=float)
            p2 = np.asarray(self.project_vertex(v2), dtype=float)
            lengths.append(float(np.linalg.norm(p1 - p2)))
        if not lengths:
            raise ValueError("Board has no edges to infer a screen-space scale.")
        return float(np.median(lengths))

    def _nearest(
        self,
        candidates: dict[object, Point],
        point: Point,
        max_distance: float | None,
    ):
        target = np.asarray(point, dtype=float)
        best_key = None
        best_distance = None
        for key, candidate in candidates.items():
            distance = float(np.linalg.norm(np.asarray(candidate, dtype=float) - target))
            if best_distance is None or distance < best_distance:
                best_key = key
                best_distance = distance
        if best_key is None:
            return None
        if max_distance is not None and best_distance is not None and best_distance > max_distance:
            return None
        return best_key

    def nearest_hex(self, point: Point, max_distance: float | None = None):
        threshold = max_distance if max_distance is not None else self.infer_scale() * 0.9
        return self._nearest(self.hex_centers(), point, threshold)

    def nearest_vertex(self, point: Point, max_distance: float | None = None):
        threshold = max_distance if max_distance is not None else self.infer_scale() * 0.4
        return self._nearest(self.vertices(), point, threshold)

    def nearest_edge(self, point: Point, max_distance: float | None = None):
        threshold = max_distance if max_distance is not None else self.infer_scale() * 0.45
        return self._nearest(self.edge_midpoints(), point, threshold)
