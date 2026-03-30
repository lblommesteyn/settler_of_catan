"""End-to-end live opening assistant for Colonist screenshots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from ..features.opening_features import compute_all_vertex_features, compute_opening_features, identify_archetype
from ..models.heuristic import WeightedHeuristic
from .bootstrap import AutoBoardBootstrap, auto_bootstrap_board
from .detector import DefaultPlayerPalette, DetectionError, PrototypeColorClassifier
from .ocr import OCRUnavailableError, read_text
from .runtime import ScreenRegion, grab_screen
from .schema import PlayerColor, PublicStructures


class PromptKind(str, Enum):
    PLACE_SETTLEMENT = "place_settlement"
    PLACE_ROAD = "place_road"
    OTHER = "other"


@dataclass(frozen=True)
class ScreenPrompt:
    text: str
    kind: PromptKind
    bbox: Optional[tuple[int, int, int, int]]
    my_color: Optional[PlayerColor]


@dataclass(frozen=True)
class OpeningSuggestion:
    kind: PromptKind
    score: float
    summary: str
    plan: Optional[str] = None
    vertex: Optional[tuple[float, float]] = None
    edge: Optional[frozenset] = None


@dataclass(frozen=True)
class OpeningScreenAnalysis:
    bootstrap: AutoBoardBootstrap
    prompt: ScreenPrompt
    public_players: tuple[PublicStructures, ...]
    my_color: PlayerColor
    seat: int
    suggestions: tuple[OpeningSuggestion, ...]
    annotated_frame: np.ndarray


def _resource_label(resource) -> str:
    return {
        "wood": "wood",
        "brick": "brick",
        "sheep": "sheep",
        "wheat": "wheat",
        "ore": "ore",
        "desert": "desert",
    }[resource.value]


def describe_vertex(board, vertex: tuple[float, float]) -> str:
    tiles = sorted(
        board.vertex_hexes.get(vertex, []),
        key=lambda tile: (tile.number is None, -(tile.number or 0), _resource_label(tile.resource)),
    )
    parts = []
    for tile in tiles:
        label = _resource_label(tile.resource)
        parts.append(f"{label} {tile.number}" if tile.number is not None else label)
    return " / ".join(parts) if parts else str(vertex)


def describe_edge(board, edge: frozenset) -> str:
    v1, v2 = tuple(edge)
    return f"{describe_vertex(board, v1)} -> {describe_vertex(board, v2)}"


def click_point(calibration, vertex: tuple[float, float]) -> tuple[int, int]:
    point = calibration.project_vertex(vertex)
    return int(round(point[0])), int(round(point[1]))


def _suggestion_target_text(bootstrap: AutoBoardBootstrap, suggestion: OpeningSuggestion) -> str:
    if suggestion.vertex is not None:
        x, y = click_point(bootstrap.calibration, suggestion.vertex)
        return f"Click {describe_vertex(bootstrap.board, suggestion.vertex)} @ ({x}, {y})"
    if suggestion.edge is not None:
        v1, v2 = tuple(suggestion.edge)
        x1, y1 = click_point(bootstrap.calibration, v1)
        x2, y2 = click_point(bootstrap.calibration, v2)
        return f"Drag road {describe_edge(bootstrap.board, suggestion.edge)} @ ({x1}, {y1}) -> ({x2}, {y2})"
    return "unknown target"


def suggestion_target_text(analysis: OpeningScreenAnalysis, suggestion: OpeningSuggestion) -> str:
    return _suggestion_target_text(analysis.bootstrap, suggestion)


def _dominant_resource(features) -> str:
    pip_by_resource = {
        "wood": features.combined_wood_pips,
        "brick": features.combined_brick_pips,
        "sheep": features.combined_sheep_pips,
        "wheat": features.combined_wheat_pips,
        "ore": features.combined_ore_pips,
    }
    return max(pip_by_resource.items(), key=lambda item: item[1])[0]


def _opening_plan(features) -> str:
    archetype_key = identify_archetype(features)
    archetype_label = {
        "ore_wheat": "Ore/Wheat Engine",
        "road_race": "Road Race",
        "port_engine": "Port Engine",
        "balanced": "Balanced Expansion",
        "high_pip": "High Production",
    }[archetype_key]
    dominant_resource = _dominant_resource(features)

    if archetype_key == "ore_wheat":
        buy = "city > dev > city; only spend on roads to lock a clean third spot"
    elif archetype_key == "road_race" or (
        features.expansion_vertex_count >= 7 and features.combined_settlement_pips >= 8
    ):
        buy = "road -> settlement -> road; delay devs until the expansion race settles"
    elif archetype_key == "port_engine":
        buy = f"reach the port and cash excess {dominant_resource} into ore/wheat for cities or devs"
    elif features.combined_city_pips >= 8:
        buy = "city first if the ore/wheat comes in, then dev or settlement depending the board"
    else:
        buy = "take the fastest spend each turn; usually road/settlement first, then city"

    if archetype_key == "road_race":
        lean = "Use early wood/brick tempo to own lanes and deny space."
    elif archetype_key == "port_engine":
        lean = f"Trade your strongest resource, {dominant_resource}, instead of waiting on perfect rolls."
    elif archetype_key == "high_pip":
        lean = "Spend raw production quickly before robber pressure catches up."
    elif archetype_key == "ore_wheat":
        lean = "Play for early cities, stable dev pressure, and a strong late game."
    else:
        lean = "Stay flexible and pivot into the fastest open line."

    return f"Lean: {archetype_label}. {lean} Buy: {buy}."


def _sample_region_mean(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.shape[1], x1)
    y1 = min(image.shape[0], y1)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(3, dtype=float)
    return patch.reshape(-1, 3).mean(axis=0)


def _detect_prompt(image: np.ndarray) -> ScreenPrompt:
    height, width = image.shape[:2]
    crop = image[int(height * 0.82) : height, int(width * 0.25) : int(width * 0.75)]
    try:
        results = read_text(crop, detail=1, paragraph=False)
    except OCRUnavailableError as exc:
        raise DetectionError(str(exc)) from exc

    best_prompt = None
    best_bbox = None
    for result in results:
        if len(result) < 2:
            continue
        bbox, raw_text = result[0], str(result[1])
        normalized = " ".join(raw_text.lower().split())
        if "place settlement" in normalized:
            best_prompt = (raw_text, PromptKind.PLACE_SETTLEMENT)
            best_bbox = bbox
            break
        if "place road" in normalized:
            best_prompt = (raw_text, PromptKind.PLACE_ROAD)
            best_bbox = bbox
            break

    if best_prompt is None or best_bbox is None:
        return ScreenPrompt(text="", kind=PromptKind.OTHER, bbox=None, my_color=None)

    xs = [point[0] for point in best_bbox]
    ys = [point[1] for point in best_bbox]
    bbox_abs = (
        int(min(xs) + width * 0.25),
        int(min(ys) + height * 0.82),
        int(max(xs) + width * 0.25),
        int(max(ys) + height * 0.82),
    )
    prompt_text, prompt_kind = best_prompt
    return ScreenPrompt(
        text=prompt_text,
        kind=prompt_kind,
        bbox=bbox_abs,
        my_color=None,
    )


def _infer_prompt_color(image: np.ndarray, prompt: ScreenPrompt) -> Optional[PlayerColor]:
    if prompt.bbox is None:
        return None
    x0, y0, _x1, y1 = prompt.bbox
    icon_size = max(18, y1 - y0)
    left = x0 - int(icon_size * 1.3)
    top = y0 - icon_size // 3
    right = x0 - int(icon_size * 0.2)
    bottom = y1 + icon_size // 3
    patch = image[max(0, top) : min(image.shape[0], bottom), max(0, left) : min(image.shape[1], right)]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturated = patch[hsv[:, :, 1] > 100]
    sample = (
        saturated.reshape(-1, 3).mean(axis=0)
        if saturated.size
        else _sample_region_mean(
            image,
            left,
            top,
            right,
            bottom,
        )
    )
    classifier = PrototypeColorClassifier(
        {
            PlayerColor.RED.value: DefaultPlayerPalette[PlayerColor.RED.value],
            PlayerColor.BLUE.value: DefaultPlayerPalette[PlayerColor.BLUE.value],
            PlayerColor.ORANGE.value: DefaultPlayerPalette[PlayerColor.ORANGE.value],
            PlayerColor.GREEN.value: DefaultPlayerPalette[PlayerColor.GREEN.value],
        }
    )
    result = classifier.classify(sample)
    if result.label is None:
        return None
    return PlayerColor.from_value(result.label)


def _detect_public_players(image: np.ndarray, bootstrap: AutoBoardBootstrap) -> tuple[PublicStructures, ...]:
    calibration = bootstrap.calibration
    board = bootstrap.board
    scale = calibration.infer_scale()
    masks = _opening_color_masks(image, bootstrap)
    color_order = (
        PlayerColor.RED,
        PlayerColor.BLUE,
        PlayerColor.ORANGE,
        PlayerColor.GREEN,
    )
    public_players: list[PublicStructures] = []
    for player_id, color in enumerate(color_order):
        mask = masks[color]
        components = _mask_components(mask)
        settlement_scores = _vertex_component_scores(components, calibration, board, scale)
        settlements = _select_settlements(settlement_scores, board, max_count=1)
        roads = _select_roads_for_settlements(components, settlements, calibration, board, scale)
        public_players.append(
            PublicStructures(
                player_id=player_id,
                color=color,
                settlements=frozenset(settlements),
                roads=frozenset(roads),
            )
        )
    return tuple(public_players)


def _opening_color_masks(image: np.ndarray, bootstrap: AutoBoardBootstrap) -> dict[PlayerColor, np.ndarray]:
    x_coords = np.arange(image.shape[1])[None, :]
    masks = {
        PlayerColor.RED: (
            (image[:, :, 0] > 170)
            & (image[:, :, 1] < 120)
            & (image[:, :, 2] < 130)
        ).astype(np.uint8)
        * 255,
        PlayerColor.ORANGE: (
            (image[:, :, 0] > 170)
            & (image[:, :, 1] > 110)
            & (image[:, :, 1] < 210)
            & (image[:, :, 2] < 130)
        ).astype(np.uint8)
        * 255,
        PlayerColor.GREEN: (
            (image[:, :, 1] > 180)
            & (image[:, :, 0] < 140)
            & (image[:, :, 2] < 120)
        ).astype(np.uint8)
        * 255,
        PlayerColor.BLUE: (
            (image[:, :, 2] > 210)
            & (image[:, :, 1] > 70)
            & (image[:, :, 1] < 150)
            & (image[:, :, 0] < 120)
            & (x_coords > 300)
        ).astype(np.uint8)
        * 255,
    }
    scale = bootstrap.calibration.infer_scale()
    interior_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hex_points = np.asarray(list(bootstrap.calibration.hex_centers().values()), dtype=float)
    x_min = int(np.floor(hex_points[:, 0].min() - scale))
    x_max = int(np.ceil(hex_points[:, 0].max() + scale))
    y_min = int(np.floor(hex_points[:, 1].min() - scale))
    y_max = int(np.ceil(hex_points[:, 1].max() + scale))
    for center in hex_points:
        cv2.circle(
            interior_mask,
            (int(round(center[0])), int(round(center[1]))),
            int(round(scale * 0.38)),
            255,
            -1,
        )
    for color, mask in masks.items():
        mask = mask.copy()
        mask[interior_mask > 0] = 0
        clipped = np.zeros_like(mask)
        clipped[max(0, y_min) : min(mask.shape[0], y_max), max(0, x_min) : min(mask.shape[1], x_max)] = (
            mask[max(0, y_min) : min(mask.shape[0], y_max), max(0, x_min) : min(mask.shape[1], x_max)]
        )
        masks[color] = clipped
    return masks


def _mask_components(mask: np.ndarray) -> tuple[tuple[int, np.ndarray], ...]:
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[int, np.ndarray]] = []
    for index in range(1, num_labels):
        x, y, width, height, area = stats[index]
        if not (20 <= area <= 5000):
            continue
        if y < 250:
            continue
        components.append((int(area), np.asarray(centroids[index], dtype=float)))
    return tuple(components)


def _vertex_component_scores(
    components: tuple[tuple[int, np.ndarray], ...],
    calibration,
    board,
    scale: float,
) -> list[tuple[float, tuple[float, float]]]:
    scores: list[tuple[float, tuple[float, float]]] = []
    threshold = scale * 0.35
    for vertex in board.all_vertices():
        point = np.asarray(calibration.project_vertex(vertex), dtype=float)
        score = 0.0
        for area, centroid in components:
            distance = float(np.linalg.norm(centroid - point))
            if distance <= threshold:
                score += area * (1.0 - distance / threshold)
        if score > 10.0:
            scores.append((score, vertex))
    scores.sort(reverse=True, key=lambda item: item[0])
    return scores


def _select_settlements(
    settlement_scores: list[tuple[float, tuple[float, float]]],
    board,
    max_count: int,
) -> tuple[tuple[float, float], ...]:
    chosen: list[tuple[float, float]] = []
    for score, vertex in settlement_scores:
        if score < 35.0:
            continue
        if vertex in chosen:
            continue
        if any(vertex in board.graph.vertex_neighbors[other] or vertex == other for other in chosen):
            continue
        chosen.append(vertex)
        if len(chosen) >= max_count:
            break
    return tuple(chosen)


def _edge_component_score(
    components: tuple[tuple[int, np.ndarray], ...],
    midpoint: np.ndarray,
    scale: float,
) -> float:
    threshold = scale * 0.30
    score = 0.0
    for area, centroid in components:
        distance = float(np.linalg.norm(centroid - midpoint))
        if distance <= threshold:
            score += area * (1.0 - distance / threshold)
    return score


def _select_roads_for_settlements(
    components: tuple[tuple[int, np.ndarray], ...],
    settlements: tuple[tuple[float, float], ...],
    calibration,
    board,
    scale: float,
) -> tuple[frozenset, ...]:
    roads: list[frozenset] = []
    used_edges: set[frozenset] = set()
    for settlement in settlements:
        best_score = 0.0
        best_edge = None
        for neighbor in board.graph.vertex_neighbors[settlement]:
            edge = frozenset({settlement, neighbor})
            if edge in used_edges:
                continue
            midpoint = np.asarray(calibration.project_edge_midpoint(edge), dtype=float)
            score = _edge_component_score(components, midpoint, scale)
            if score > best_score:
                best_score = score
                best_edge = edge
        if best_edge is not None and best_score >= 5.0:
            roads.append(best_edge)
            used_edges.add(best_edge)
    return tuple(roads)


def _occupied_vertices(public_players: tuple[PublicStructures, ...]) -> set[tuple[float, float]]:
    occupied: set[tuple[float, float]] = set()
    for player in public_players:
        occupied.update(player.settlements)
        occupied.update(player.cities)
    return occupied


def _occupied_edges(public_players: tuple[PublicStructures, ...]) -> set[frozenset]:
    occupied: set[frozenset] = set()
    for player in public_players:
        occupied.update(player.roads)
    return occupied


def _normalize_setup_public_players(
    public_players: tuple[PublicStructures, ...],
    my_color: PlayerColor,
    prompt_kind: PromptKind,
) -> tuple[PublicStructures, ...]:
    if prompt_kind != PromptKind.PLACE_SETTLEMENT:
        return public_players
    normalized: list[PublicStructures] = []
    others_with_settlement = sum(
        1 for player in public_players if player.color != my_color and len(player.settlements) > 0
    )
    for player in public_players:
        if player.color == my_color and others_with_settlement >= 3 and len(player.settlements) > 0:
            normalized.append(
                PublicStructures(
                    player_id=player.player_id,
                    color=player.color,
                    settlements=frozenset(),
                    roads=frozenset(),
                )
            )
        else:
            normalized.append(player)
    return tuple(normalized)


def _resolve_auto_color_from_setup_state(
    public_players: tuple[PublicStructures, ...],
    guessed_color: Optional[PlayerColor],
    prompt_kind: PromptKind,
) -> Optional[PlayerColor]:
    if prompt_kind == PromptKind.PLACE_SETTLEMENT:
        zero_settlement_colors = [player.color for player in public_players if player.color is not None and len(player.settlements) == 0]
        if len(zero_settlement_colors) == 1:
            if guessed_color is None:
                return zero_settlement_colors[0]
            guessed_player = next((player for player in public_players if player.color == guessed_color), None)
            if guessed_player is None or len(guessed_player.settlements) > 0:
                return zero_settlement_colors[0]
    return guessed_color


def _legal_opening_vertices(board, occupied: set[tuple[float, float]]) -> list[tuple[float, float]]:
    blocked = set(occupied)
    for vertex in occupied:
        blocked.update(board.graph.vertex_neighbors[vertex])
    return [vertex for vertex in board.legal_starting_vertices() if vertex not in blocked]


def _infer_seat(
    prompt_kind: PromptKind,
    public_players: tuple[PublicStructures, ...],
    my_color: PlayerColor,
) -> int:
    by_color = {player.color: player for player in public_players if player.color is not None}
    mine = by_color.get(my_color)
    if mine is None:
        raise DetectionError(f"Could not find public structures for local color '{my_color.value}'.")

    my_settlements = len(mine.settlements)
    if prompt_kind == PromptKind.PLACE_SETTLEMENT:
        if my_settlements == 0:
            return sum(1 for player in public_players if player.color != my_color and len(player.settlements) >= 1)
        if my_settlements == 1:
            return 3 - sum(1 for player in public_players if player.color != my_color and len(player.settlements) >= 2)
    if prompt_kind == PromptKind.PLACE_ROAD:
        if my_settlements == 1:
            return sum(1 for player in public_players if player.color != my_color and len(player.settlements) >= 1)
        if my_settlements == 2:
            return 3 - sum(1 for player in public_players if player.color != my_color and len(player.settlements) >= 2)
    raise DetectionError("The live opening assistant could not infer the current setup seat from the visible board.")


def _best_future_pair_score(
    board,
    first_vertex: tuple[float, float],
    occupied: set[tuple[float, float]],
    seat: int,
    vf_cache,
    model: WeightedHeuristic,
) -> tuple[float, Optional[tuple[float, float]]]:
    future_legal = _legal_opening_vertices(board, occupied | {first_vertex})
    best_score = float("-inf")
    best_vertex = None
    for second_vertex in future_legal:
        features = compute_opening_features(first_vertex, second_vertex, seat, board, vf_cache)
        score = model.predict_win_probability(features)
        if score > best_score:
            best_score = score
            best_vertex = second_vertex
    return best_score, best_vertex


def _suggest_settlements(
    bootstrap: AutoBoardBootstrap,
    public_players: tuple[PublicStructures, ...],
    my_color: PlayerColor,
    seat: int,
    top_k: int,
) -> tuple[OpeningSuggestion, ...]:
    board = bootstrap.board
    occupied = _occupied_vertices(public_players)
    vf_cache = compute_all_vertex_features(board)
    model = WeightedHeuristic()
    by_color = {player.color: player for player in public_players if player.color is not None}
    mine = by_color[my_color]

    suggestions: list[OpeningSuggestion] = []
    if len(mine.settlements) == 0:
        for vertex in _legal_opening_vertices(board, occupied):
            pair_score, follow_up = _best_future_pair_score(board, vertex, occupied, seat, vf_cache, model)
            if follow_up is None:
                continue
            suggestions.append(
                OpeningSuggestion(
                    kind=PromptKind.PLACE_SETTLEMENT,
                    vertex=vertex,
                    score=pair_score,
                    summary=(
                        f"First setup settlement. Best paired follow-up is {describe_vertex(board, follow_up)} "
                        f"for an estimated {pair_score * 100:.1f}% opening score."
                    ),
                    plan=_opening_plan(compute_opening_features(vertex, follow_up, seat, board, vf_cache)),
                )
            )
    else:
        first_vertex = next(iter(mine.settlements))
        for vertex in _legal_opening_vertices(board, occupied):
            features = compute_opening_features(first_vertex, vertex, seat, board, vf_cache)
            score = model.predict_win_probability(features)
            suggestions.append(
                OpeningSuggestion(
                    kind=PromptKind.PLACE_SETTLEMENT,
                    vertex=vertex,
                    score=score,
                    summary=(
                        f"Second setup settlement after {describe_vertex(board, first_vertex)}. "
                        f"Estimated opening score: {score * 100:.1f}%."
                    ),
                    plan=_opening_plan(features),
                )
            )
    suggestions.sort(key=lambda item: item.score, reverse=True)
    return tuple(suggestions[:top_k])


def _road_anchor(mine: PublicStructures, board) -> tuple[float, float]:
    if len(mine.settlements) == 1:
        return next(iter(mine.settlements))
    incident_counts = {
        settlement: sum(1 for edge in mine.roads if settlement in edge)
        for settlement in mine.settlements
    }
    anchor = next((vertex for vertex, count in incident_counts.items() if count == 0), None)
    if anchor is not None:
        return anchor
    return max(mine.settlements, key=board.pip_count)


def _road_future_score(
    board,
    anchor: tuple[float, float],
    next_vertex: tuple[float, float],
    occupied_vertices: set[tuple[float, float]],
    vf_cache,
) -> tuple[float, Optional[tuple[float, float]]]:
    best_target = None
    best_score = float("-inf")
    blocked = set(occupied_vertices)
    for target in board.graph.vertex_neighbors[next_vertex]:
        if target == anchor:
            continue
        if target in blocked:
            continue
        if any(neighbor in blocked for neighbor in board.graph.vertex_neighbors[target]):
            continue
        features = vf_cache[target]
        score = (
            features.total_pips
            + 1.5 * features.resource_count
            + 1.0 * float(features.has_port)
            + 2.0 * features.resource_entropy
        )
        if score > best_score:
            best_score = score
            best_target = target
    return best_score, best_target


def _suggest_roads(
    bootstrap: AutoBoardBootstrap,
    public_players: tuple[PublicStructures, ...],
    my_color: PlayerColor,
    seat: int,
    top_k: int,
) -> tuple[OpeningSuggestion, ...]:
    board = bootstrap.board
    occupied_vertices = _occupied_vertices(public_players)
    occupied_edges = _occupied_edges(public_players)
    vf_cache = compute_all_vertex_features(board)
    by_color = {player.color: player for player in public_players if player.color is not None}
    mine = by_color[my_color]
    anchor = _road_anchor(mine, board)
    current_pair_features = None
    if len(mine.settlements) >= 2:
        v1, v2 = tuple(sorted(mine.settlements))
        current_pair_features = compute_opening_features(v1, v2, seat, board, vf_cache)

    suggestions: list[OpeningSuggestion] = []
    for neighbor in board.graph.vertex_neighbors[anchor]:
        edge = frozenset({anchor, neighbor})
        if edge in occupied_edges:
            continue
        score, target = _road_future_score(board, anchor, neighbor, occupied_vertices, vf_cache)
        plan = _opening_plan(current_pair_features) if current_pair_features is not None else None
        if plan is None and target is not None:
            plan = _opening_plan(compute_opening_features(anchor, target, seat, board, vf_cache))
        summary = (
            f"Setup road from {describe_vertex(board, anchor)} toward {describe_vertex(board, neighbor)}. "
            f"Best follow-up settlement lane: {describe_vertex(board, target)}."
            if target is not None
            else f"Setup road from {describe_vertex(board, anchor)} toward {describe_vertex(board, neighbor)}."
        )
        suggestions.append(
            OpeningSuggestion(
                kind=PromptKind.PLACE_ROAD,
                edge=edge,
                score=score,
                summary=summary,
                plan=plan,
            )
        )
    suggestions.sort(key=lambda item: item.score, reverse=True)
    return tuple(suggestions[:top_k])


def _draw_opening_overlay(
    image: np.ndarray,
    bootstrap: AutoBoardBootstrap,
    prompt: ScreenPrompt,
    suggestions: tuple[OpeningSuggestion, ...],
    seat: int,
    my_color: PlayerColor,
) -> np.ndarray:
    calibration = bootstrap.calibration
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    palette = [(46, 204, 113), (39, 174, 96), (241, 196, 15)]
    for index, suggestion in enumerate(suggestions, start=1):
        color = palette[min(index - 1, len(palette) - 1)]
        if suggestion.vertex is not None:
            point = calibration.project_vertex(suggestion.vertex)
            cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 16, color, 3)
            cv2.putText(
                overlay,
                str(index),
                (int(round(point[0])) + 18, int(round(point[1])) - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )
        elif suggestion.edge is not None:
            v1, v2 = tuple(suggestion.edge)
            p1 = calibration.project_vertex(v1)
            p2 = calibration.project_vertex(v2)
            cv2.line(
                overlay,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                color,
                5,
                cv2.LINE_AA,
            )
            midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            cv2.putText(
                overlay,
                str(index),
                (int(round(midpoint[0])) + 12, int(round(midpoint[1])) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

    def _fit_text(text: str, max_chars: int = 96) -> str:
        return text if len(text) <= max_chars else f"{text[: max_chars - 3]}..."

    title = prompt.text or "Opening Assistant"
    header = f"{title} | {my_color.value} | S{seat + 1}"
    legend_lines = [header]
    for index, suggestion in enumerate(suggestions, start=1):
        action = _fit_text(_suggestion_target_text(bootstrap, suggestion))
        detail = _fit_text(suggestion.summary, max_chars=104)
        plan = _fit_text(suggestion.plan, max_chars=104) if suggestion.plan else None
        legend_lines.append(f"{index}. {action}")
        legend_lines.append(f"   {detail}")
        if plan is not None:
            legend_lines.append(f"   {plan}")

    line_height = 28
    legend_height = 28 + line_height * len(legend_lines)
    legend_width = min(overlay.shape[1] - 40, 1480)
    cv2.rectangle(overlay, (20, 20), (20 + legend_width, 20 + legend_height), (10, 24, 38), -1)
    for line_index, line in enumerate(legend_lines):
        font_scale = 0.78 if line_index == 0 else 0.62
        thickness = 2 if line_index == 0 else 1
        color = (245, 245, 245) if line_index == 0 else (228, 235, 239)
        y = 48 + line_index * line_height
        cv2.putText(
            overlay,
            line,
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def analyze_opening_screen(
    image: np.ndarray,
    *,
    my_color: Optional[PlayerColor] = None,
    top_k: int = 3,
    bootstrap: Optional[AutoBoardBootstrap] = None,
) -> OpeningScreenAnalysis:
    bootstrap = bootstrap or auto_bootstrap_board(image)
    prompt = _detect_prompt(image)
    if prompt.kind not in (PromptKind.PLACE_SETTLEMENT, PromptKind.PLACE_ROAD):
        raise DetectionError("The current Colonist screen does not show an opening placement prompt.")

    public_players = _detect_public_players(image, bootstrap)
    inferred_color = my_color or _infer_prompt_color(image, prompt)
    inferred_color = _resolve_auto_color_from_setup_state(public_players, inferred_color, prompt.kind)
    if inferred_color is None:
        raise DetectionError("Could not infer the local player color from the active Colonist prompt.")
    prompt = ScreenPrompt(text=prompt.text, kind=prompt.kind, bbox=prompt.bbox, my_color=inferred_color)
    public_players = _normalize_setup_public_players(public_players, inferred_color, prompt.kind)
    seat = _infer_seat(prompt.kind, public_players, inferred_color)
    if prompt.kind == PromptKind.PLACE_SETTLEMENT:
        suggestions = _suggest_settlements(bootstrap, public_players, inferred_color, seat, top_k=top_k)
    else:
        suggestions = _suggest_roads(bootstrap, public_players, inferred_color, seat, top_k=top_k)
    annotated = _draw_opening_overlay(image, bootstrap, prompt, suggestions, seat, inferred_color)
    return OpeningScreenAnalysis(
        bootstrap=bootstrap,
        prompt=prompt,
        public_players=public_players,
        my_color=inferred_color,
        seat=seat,
        suggestions=suggestions,
        annotated_frame=annotated,
    )


def _analysis_fingerprint(analysis: OpeningScreenAnalysis) -> tuple:
    player_payload = []
    for player in analysis.public_players:
        player_payload.append(
            (
                player.color.value if player.color is not None else None,
                tuple(sorted(player.settlements)),
                tuple(sorted(player.cities)),
                tuple(sorted(sorted(tuple(vertex) for vertex in edge) for edge in player.roads)),
            )
        )
    suggestion_payload = []
    for suggestion in analysis.suggestions:
        suggestion_payload.append(
            (
                suggestion.kind.value,
                suggestion.vertex,
                tuple(sorted(suggestion.edge)) if suggestion.edge is not None else None,
                round(suggestion.score, 4),
            )
        )
    return (
        analysis.prompt.kind.value,
        analysis.prompt.my_color.value,
        analysis.seat,
        tuple(player_payload),
        tuple(suggestion_payload),
    )


class OpeningLiveRunner:
    """Continuous live opening assistant driven directly by the screen."""

    def run(
        self,
        *,
        region: Optional[ScreenRegion] = None,
        my_color: Optional[PlayerColor] = None,
        top_k: int = 3,
        interval_s: float = 1.0,
        once: bool = False,
        show: bool = True,
    ) -> None:
        previous = None
        bootstrap: Optional[AutoBoardBootstrap] = None
        window_name = "Catan Opening Assistant"
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                frame = grab_screen(region)
                try:
                    analysis = analyze_opening_screen(frame, my_color=my_color, top_k=top_k, bootstrap=bootstrap)
                except DetectionError:
                    bootstrap = None
                    analysis = analyze_opening_screen(frame, my_color=my_color, top_k=top_k, bootstrap=None)
                bootstrap = analysis.bootstrap
                fingerprint = _analysis_fingerprint(analysis)
                if fingerprint != previous:
                    print(
                        f"\n[{time.strftime('%H:%M:%S')}] "
                        f"{analysis.prompt.kind.value} color={analysis.my_color.value} seat=S{analysis.seat + 1}"
                    )
                    for index, suggestion in enumerate(analysis.suggestions, start=1):
                        print(f"{index}. {suggestion.score:.3f} -> {suggestion_target_text(analysis, suggestion)}")
                        print(f"   {suggestion.summary}")
                        if suggestion.plan:
                            print(f"   {suggestion.plan}")
                    previous = fingerprint
                if show:
                    bgr = cv2.cvtColor(analysis.annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, bgr)
                    key = cv2.waitKey(max(1, int(interval_s * 1000))) & 0xFF
                    if key == ord("q"):
                        return
                else:
                    time.sleep(interval_s)
                if once:
                    return
        finally:
            if show:
                cv2.destroyWindow(window_name)
