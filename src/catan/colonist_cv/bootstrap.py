"""Automatic board bootstrap from Colonist screenshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..board.board import CatanBoard, NUMBER_POOL, PortType, Resource
from ..board.hex_grid import AXIAL_POSITIONS
from .detector import DetectionError
from .geometry import BoardCalibration
from .ocr import OCRUnavailableError, read_text


VALID_NUMBER_TOKENS = {str(value) for value in NUMBER_POOL}
RESOURCE_HSV_PROTOTYPES: dict[Resource, tuple[float, float, float]] = {
    Resource.BRICK: (14.0, 190.0, 187.0),
    Resource.WOOD: (54.0, 135.0, 145.0),
    Resource.SHEEP: (35.0, 185.0, 170.0),
    Resource.WHEAT: (24.0, 175.0, 205.0),
    Resource.ORE: (30.0, 45.0, 170.0),
    Resource.DESERT: (24.0, 100.0, 199.0),
}


@dataclass(frozen=True)
class TokenCandidate:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: int
    width: int
    height: int


@dataclass(frozen=True)
class AutoBoardBootstrap:
    """Parsed board geometry and layout inferred directly from the screen."""

    board: CatanBoard
    calibration: BoardCalibration
    desert_hex: tuple[int, int]
    token_candidates: tuple[TokenCandidate, ...]
    token_centers: dict[tuple[int, int], tuple[float, float]]
    numbers_by_hex: dict[tuple[int, int], Optional[int]]
    resources_by_hex: dict[tuple[int, int], Resource]
    ports_are_conservative_generic: bool = True


@dataclass(frozen=True)
class _LatticeFit:
    desert_hex: tuple[int, int]
    dx: float
    dy: float
    tx: float
    ty: float
    matches: tuple[tuple[int, int], ...]

    def project_hex(self, hex_coord: tuple[int, int]) -> tuple[float, float]:
        q, r = hex_coord
        return (self.tx + self.dx * (q + 0.5 * r), self.ty + self.dy * r)


def _token_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] < 96) & (hsv[:, :, 2] > 168)
    return mask.astype(np.uint8) * 255


def _extract_token_candidates(image: np.ndarray) -> tuple[TokenCandidate, ...]:
    mask = _token_mask(image)
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    strict: list[TokenCandidate] = []
    relaxed: list[TokenCandidate] = []
    for index in range(1, num_labels):
        x, y, width, height, area = stats[index]
        aspect = width / max(height, 1)
        if not (0.7 <= aspect <= 1.35):
            continue
        if width < 18 or height < 18:
            continue
        candidate = TokenCandidate(
            center=(float(centroids[index][0]), float(centroids[index][1])),
            bbox=(int(x), int(y), int(width), int(height)),
            area=int(area),
            width=int(width),
            height=int(height),
        )
        if 1700 <= area <= 3200 and 48 <= width <= 62 and 48 <= height <= 62:
            strict.append(candidate)
        elif 900 <= area <= 4200 and 30 <= width <= 70 and 30 <= height <= 70:
            relaxed.append(candidate)
    if len(strict) >= 12:
        return tuple(strict)
    return tuple(relaxed)


def _largest_dense_cluster(candidates: tuple[TokenCandidate, ...]) -> tuple[TokenCandidate, ...]:
    if not candidates:
        raise DetectionError("Could not find any number-token candidates on the screen.")

    centers = np.asarray([candidate.center for candidate in candidates], dtype=float)
    side = float(np.median([(candidate.width + candidate.height) / 2.0 for candidate in candidates]))
    max_distance = max(110.0, side * 4.4)
    adjacency = [set() for _ in candidates]
    for left in range(len(candidates)):
        deltas = centers[left + 1 :] - centers[left]
        if deltas.size == 0:
            continue
        distances = np.linalg.norm(deltas, axis=1)
        for offset, distance in enumerate(distances, start=left + 1):
            if distance <= max_distance:
                adjacency[left].add(offset)
                adjacency[offset].add(left)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(candidates)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    best = max(components, key=len)
    cluster = tuple(candidates[index] for index in best)
    if len(cluster) < 12:
        raise DetectionError(
            "Found token-like boxes, but not enough of them to fit a Colonist board reliably."
        )
    return cluster


def _greedy_matches(
    predicted: np.ndarray,
    observed: np.ndarray,
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    pair_candidates: list[tuple[float, int, int]] = []
    distances = np.linalg.norm(predicted[:, None, :] - observed[None, :, :], axis=2)
    for pred_index in range(predicted.shape[0]):
        obs_index = int(np.argmin(distances[pred_index]))
        distance = float(distances[pred_index, obs_index])
        if distance <= threshold:
            pair_candidates.append((distance, pred_index, obs_index))
    pair_candidates.sort()

    used_pred: set[int] = set()
    used_obs: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _distance, pred_index, obs_index in pair_candidates:
        if pred_index in used_pred or obs_index in used_obs:
            continue
        used_pred.add(pred_index)
        used_obs.add(obs_index)
        matches.append((pred_index, obs_index))
    return tuple(matches)


def _fit_linear_parameters(
    coordinates: np.ndarray,
    observed: np.ndarray,
    matches: tuple[tuple[int, int], ...],
) -> tuple[float, float, float, float]:
    if not matches:
        raise DetectionError("Could not align the detected token lattice to canonical hex coordinates.")

    u_features = np.asarray([[coordinates[pred_index, 0], 1.0] for pred_index, _ in matches], dtype=float)
    r_features = np.asarray([[coordinates[pred_index, 1], 1.0] for pred_index, _ in matches], dtype=float)
    x_targets = np.asarray([observed[obs_index, 0] for _, obs_index in matches], dtype=float)
    y_targets = np.asarray([observed[obs_index, 1] for _, obs_index in matches], dtype=float)

    dx, tx = np.linalg.lstsq(u_features, x_targets, rcond=None)[0]
    dy, ty = np.linalg.lstsq(r_features, y_targets, rcond=None)[0]
    return float(dx), float(dy), float(tx), float(ty)


def _fit_hex_lattice(cluster: tuple[TokenCandidate, ...]) -> _LatticeFit:
    observed = np.asarray([candidate.center for candidate in cluster], dtype=float)
    canonical = np.asarray([(q + 0.5 * r, r) for q, r in AXIAL_POSITIONS], dtype=float)

    x_span = float(observed[:, 0].max() - observed[:, 0].min())
    y_span = float(observed[:, 1].max() - observed[:, 1].min())
    dx_guess = max(x_span / 4.0, 50.0)
    dy_guess = max(y_span / 4.0, 45.0)

    best_fit: Optional[_LatticeFit] = None
    best_key: Optional[tuple[int, float]] = None

    for desert_index, desert_hex in enumerate(AXIAL_POSITIONS):
        coords = np.delete(canonical, desert_index, axis=0)
        for dx in np.linspace(dx_guess * 0.82, dx_guess * 1.18, 25):
            for dy in np.linspace(dy_guess * 0.82, dy_guess * 1.18, 25):
                predicted = np.column_stack([coords[:, 0] * dx, coords[:, 1] * dy])
                tx = float(observed[:, 0].mean() - predicted[:, 0].mean())
                ty = float(observed[:, 1].mean() - predicted[:, 1].mean())
                predicted[:, 0] += tx
                predicted[:, 1] += ty
                threshold = max(dx, dy) * 0.45
                matches = _greedy_matches(predicted, observed, threshold=threshold)
                if not matches:
                    continue
                mean_error = float(
                    np.mean(
                        [
                            np.linalg.norm(predicted[pred_index] - observed[obs_index])
                            for pred_index, obs_index in matches
                        ]
                    )
                )
                key = (len(matches), -mean_error)
                if best_key is None or key > best_key:
                    best_key = key
                    best_fit = _LatticeFit(
                        desert_hex=desert_hex,
                        dx=float(dx),
                        dy=float(dy),
                        tx=tx,
                        ty=ty,
                        matches=matches,
                    )

    if best_fit is None or best_key is None:
        raise DetectionError("Could not fit the board lattice to the detected token pattern.")

    coords = np.asarray(
        [(q + 0.5 * r, r) for q, r in AXIAL_POSITIONS if (q, r) != best_fit.desert_hex],
        dtype=float,
    )
    for _ in range(6):
        predicted = np.column_stack([coords[:, 0] * best_fit.dx + best_fit.tx, coords[:, 1] * best_fit.dy + best_fit.ty])
        threshold = max(best_fit.dx, best_fit.dy) * 0.48
        matches = _greedy_matches(predicted, observed, threshold=threshold)
        if len(matches) < 12:
            break
        dx, dy, tx, ty = _fit_linear_parameters(coords, observed, matches)
        best_fit = _LatticeFit(
            desert_hex=best_fit.desert_hex,
            dx=dx,
            dy=dy,
            tx=tx,
            ty=ty,
            matches=matches,
        )

    if len(best_fit.matches) < 12:
        raise DetectionError("The inferred hex lattice is too weak to trust for live advice.")
    return best_fit


def _predicted_token_centers(fit: _LatticeFit) -> dict[tuple[int, int], tuple[float, float]]:
    return {hex_coord: fit.project_hex(hex_coord) for hex_coord in AXIAL_POSITIONS}


def _crop_with_padding(
    image: np.ndarray,
    center: tuple[float, float],
    radius_x: int,
    radius_y: int,
) -> np.ndarray:
    cx, cy = center
    x0 = max(0, int(round(cx)) - radius_x)
    x1 = min(image.shape[1], int(round(cx)) + radius_x)
    y0 = max(0, int(round(cy)) - radius_y)
    y1 = min(image.shape[0], int(round(cy)) + radius_y)
    return image[y0:y1, x0:x1]


def _number_votes(image: np.ndarray, center: tuple[float, float], token_radius: int) -> Counter[int]:
    patch = _crop_with_padding(image, center, token_radius, token_radius)
    if patch.size == 0:
        return Counter()
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if gray.shape[0] < 20 or gray.shape[1] < 20:
        return Counter()

    pad_y = max(2, int(gray.shape[0] * 0.08))
    pad_x = max(2, int(gray.shape[1] * 0.08))
    core = gray[pad_y:-pad_y or None, pad_x:-pad_x or None]
    if core.size == 0:
        core = gray

    resized_gray = cv2.resize(core, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    resized_eq = cv2.resize(cv2.equalizeHist(core), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(resized_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverse = cv2.threshold(resized_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        resized_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    votes: Counter[int] = Counter()
    for index, variant in enumerate((resized_gray, otsu, inverse, adaptive)):
        try:
            results = read_text(variant, allowlist="0123456789", detail=0, paragraph=False)
        except OCRUnavailableError as exc:
            raise DetectionError(str(exc)) from exc
        found_valid = False
        for text in results:
            normalized = str(text).strip()
            if normalized in VALID_NUMBER_TOKENS:
                votes[int(normalized)] += 1
                found_valid = True
        if found_valid and index == 0:
            return votes
    return votes


def _assign_numbers(
    image: np.ndarray,
    token_centers: dict[tuple[int, int], tuple[float, float]],
    desert_hex: tuple[int, int],
    token_radius: int,
) -> dict[tuple[int, int], Optional[int]]:
    votes_by_hex = {
        hex_coord: _number_votes(image, center, token_radius)
        for hex_coord, center in token_centers.items()
        if hex_coord != desert_hex
    }
    remaining = Counter(NUMBER_POOL)
    assigned: dict[tuple[int, int], Optional[int]] = {}

    candidate_scores: list[tuple[int, int, tuple[int, int]]] = []
    for hex_coord, votes in votes_by_hex.items():
        for number, score in votes.items():
            candidate_scores.append((score, number, hex_coord))
    candidate_scores.sort(reverse=True)

    for score, number, hex_coord in candidate_scores:
        if hex_coord in assigned:
            continue
        if remaining[number] <= 0:
            continue
        assigned[hex_coord] = number
        remaining[number] -= 1

    for hex_coord in token_centers:
        if hex_coord == desert_hex:
            continue
        if hex_coord in assigned:
            continue
        votes = votes_by_hex[hex_coord]
        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        choice = next((number for number, _score in ranked if remaining[number] > 0), None)
        if choice is None:
            choice = next((number for number, count in remaining.items() if count > 0), None)
        if choice is None:
            raise DetectionError("Number-token OCR over-consumed the standard token pool.")
        assigned[hex_coord] = int(choice)
        remaining[choice] -= 1

    assigned[desert_hex] = None
    return assigned


def _sample_tile_signature(
    image: np.ndarray,
    center: tuple[float, float],
    scale: float,
) -> tuple[float, float, float]:
    radii = (scale * 0.50, scale * 0.66, scale * 0.80)
    angles_deg = (0, 60, 120, 180, 240, 300)
    samples: list[np.ndarray] = []
    for radius in radii:
        for angle_deg in angles_deg:
            angle = np.deg2rad(angle_deg)
            px = int(round(center[0] + radius * np.cos(angle)))
            py = int(round(center[1] + radius * np.sin(angle)))
            x0 = max(0, px - 1)
            x1 = min(image.shape[1], px + 2)
            y0 = max(0, py - 1)
            y1 = min(image.shape[0], py + 2)
            patch = image[y0:y1, x0:x1]
            if patch.size:
                samples.append(np.asarray(patch, dtype=float).reshape(-1, 3).mean(axis=0))
    if not samples:
        raise DetectionError("Could not sample any tile pixels for resource classification.")
    rgb = np.asarray(samples, dtype=float).mean(axis=0)
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _resource_distance(
    signature: tuple[float, float, float],
    prototype: tuple[float, float, float],
) -> float:
    hue, saturation, value = signature
    proto_h, proto_s, proto_v = prototype
    return (
        abs(hue - proto_h) * 3.0
        + abs(saturation - proto_s) * 1.0
        + abs(value - proto_v) * 0.5
    )


def _classify_resource(signature: tuple[float, float, float]) -> tuple[Resource, dict[Resource, float]]:
    distances = {
        resource: _resource_distance(signature, prototype)
        for resource, prototype in RESOURCE_HSV_PROTOTYPES.items()
    }
    best = min(distances, key=distances.get)
    return best, distances


def auto_bootstrap_board(image: np.ndarray) -> AutoBoardBootstrap:
    """
    Infer the visible Colonist board directly from a screenshot.

    This is aimed at live opening assistance, so ports are conservatively set
    to generic slots until a dedicated port reader is added.
    """

    cluster = _largest_dense_cluster(_extract_token_candidates(image))
    fit = _fit_hex_lattice(cluster)
    token_centers = _predicted_token_centers(fit)
    geometry_board = CatanBoard.random(seed=0)
    canonical_to_screen = np.array(
        [
            [fit.dx / np.sqrt(3.0), 0.0, fit.tx],
            [0.0, fit.dy / 1.5, fit.ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    screen_to_canonical = np.linalg.inv(canonical_to_screen)
    provisional_calibration = BoardCalibration.from_matrices(
        geometry_board,
        canonical_to_screen=canonical_to_screen,
        screen_to_canonical=screen_to_canonical,
    )
    scale = provisional_calibration.infer_scale()

    resource_distances: dict[tuple[int, int], dict[Resource, float]] = {}
    resources_by_hex: dict[tuple[int, int], Resource] = {}
    for hex_coord in AXIAL_POSITIONS:
        signature = _sample_tile_signature(image, fit.project_hex(hex_coord), scale)
        resource, distances = _classify_resource(signature)
        resources_by_hex[hex_coord] = resource
        resource_distances[hex_coord] = distances

    desert_candidates = [hex_coord for hex_coord, resource in resources_by_hex.items() if resource == Resource.DESERT]
    if desert_candidates:
        desert_hex = min(desert_candidates, key=lambda hex_coord: resource_distances[hex_coord][Resource.DESERT])
    else:
        desert_hex = min(AXIAL_POSITIONS, key=lambda hex_coord: resource_distances[hex_coord][Resource.DESERT])

    for hex_coord in AXIAL_POSITIONS:
        if hex_coord == desert_hex:
            resources_by_hex[hex_coord] = Resource.DESERT
        elif resources_by_hex[hex_coord] == Resource.DESERT:
            resources_by_hex[hex_coord] = Resource.ORE

    numbers_by_hex = _assign_numbers(
        image=image,
        token_centers=token_centers,
        desert_hex=desert_hex,
        token_radius=max(30, int(round(np.median([(candidate.width + candidate.height) / 2.0 for candidate in cluster]) * 0.72))),
    )

    resources = [resources_by_hex[hex_coord] for hex_coord in AXIAL_POSITIONS]
    numbers = [numbers_by_hex[hex_coord] for hex_coord in AXIAL_POSITIONS if hex_coord != desert_hex]
    if any(number is None for number in numbers):
        raise DetectionError("Automatic number-token parsing produced an incomplete board.")

    board = CatanBoard.from_tiles(
        AXIAL_POSITIONS,
        resources,
        [int(number) for number in numbers],
        [PortType.GENERIC] * 9,
    )
    calibration = BoardCalibration.from_matrices(
        board,
        canonical_to_screen=canonical_to_screen,
        screen_to_canonical=screen_to_canonical,
    )
    return AutoBoardBootstrap(
        board=board,
        calibration=calibration,
        desert_hex=desert_hex,
        token_candidates=cluster,
        token_centers=token_centers,
        numbers_by_hex=numbers_by_hex,
        resources_by_hex=resources_by_hex,
        ports_are_conservative_generic=True,
    )
