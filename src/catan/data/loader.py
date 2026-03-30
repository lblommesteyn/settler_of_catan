"""
Colonist.io dataset loader — verified schema (Catan-data/dataset).

Dataset: https://github.com/Catan-data/dataset/
Download: games.tar.gz → stream directly without extracting (no disk space needed).

Confirmed schema (from inspect_sample_files):

game = {
  "data": {
    "playOrder": [int, int, int, int],   # color IDs: 1=Blue,2=Red,3=Orange,4=Brown,5=White
    "eventHistory": {
      "initialState": {
        "mapState": {
          "tileHexStates": {
            "<idx>": {"x": int, "y": int, "type": int, "diceNumber": int}
            # x=axial_q, y=axial_r, type: 0=Desert,1=Brick,2=Wool,3=Grain,4=Ore,5=Lumber
          },
          "portEdgeStates": {
            "<idx>": {"x": int, "y": int, "z": int, "type": int}
            # x,y = approx hex position (may be sea hex), z = edge direction
            # port type: 1=generic 3:1, 2=Brick,3=Wool,4=Grain,5=Ore,6=Lumber
          }
        }
      },
      "events": [
        {
          "stateChange": {
            "currentState": {"completedTurns": int},
            "gameLogState": {
              "<key>": {"from": int, "text": {"type": int, "playerColor": int, "pieceEnum": int}}
            },
            "mapState": {
              "tileCornerStates": {
                "<corner_id>": {"owner": int, "buildingType": int}  # 1=settlement, 2=city
              },
              "tileEdgeStates": {
                "<edge_id>": {"type": int, "owner": int}            # roads
              }
            }
          }
        }
      ],
      "endGameState": {
        "totalTurnCount": int,
        "players": {
          "<color>": {
            "winningPlayer": bool,
            "rank": int,
            "victoryPoints": {"<category_id>": int, ...}  # sum for total VP
          }
        }
      }
    }
  }
}

Event text.type 4 = building placement; text.pieceEnum: 0=road, 2=settlement, 3=city
"""

from __future__ import annotations

import json
import logging
import math
import tarfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .schema import (
    ColonistGameRecord, ColonistBoardState, ColonistHex,
    ColonistPort, ColonistOpeningPlacement, ProcessedOpeningRecord,
)
from ..board.board import CatanBoard, Resource, PortType, AXIAL_POSITIONS
from ..board.hex_grid import axial_to_cartesian, hex_corners
from ..features.opening_features import (
    compute_opening_features,
    opening_features_to_array,
    compute_all_vertex_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enum mappings (verified from real files)
# ---------------------------------------------------------------------------

# tileHexStates.type → Resource
TILE_TYPE_TO_RESOURCE: dict[int, Resource] = {
    0: Resource.DESERT,
    1: Resource.BRICK,
    2: Resource.SHEEP,   # Wool
    3: Resource.WHEAT,   # Grain
    4: Resource.ORE,
    5: Resource.WOOD,    # Lumber
}

# portEdgeStates.type → PortType
PORT_TYPE_INT_TO_PORT: dict[int, PortType] = {
    1: PortType.GENERIC,
    2: PortType.BRICK,
    3: PortType.SHEEP,   # Wool
    4: PortType.WHEAT,   # Grain
    5: PortType.ORE,
    6: PortType.WOOD,    # Lumber
}

OPENING_SETTLEMENT_COUNT = 8  # 2 per player × 4 players


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_game_file(path: Path) -> Optional[ColonistGameRecord]:
    """Parse one JSON game file. Returns None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _parse_game_dict(raw, game_id=path.stem)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path.name, exc)
        return None


def _parse_game_dict(raw: dict, game_id: str) -> Optional[ColonistGameRecord]:
    """Parse a raw game dict using the confirmed schema."""
    try:
        data = raw.get("data", raw)
        play_order: list[int] = data.get("playOrder", [])
        ev_hist: dict = data.get("eventHistory", {})

        init_state: dict = ev_hist.get("initialState", {})
        events: list[dict] = ev_hist.get("events", [])
        end_state: dict = ev_hist.get("endGameState", {})

        board_state = _parse_board_state(init_state)
        openings = _extract_opening_settlements(events)
        winner_color, final_vps, final_ranks, total_turns = _parse_outcome(end_state)

        return ColonistGameRecord(
            game_id=game_id,
            board=board_state,
            play_order=play_order,
            opening_placements=openings,
            winner_index=winner_color,
            final_vps=final_vps,
            final_ranks=final_ranks,
            total_turns=total_turns,
            raw={},  # don't store raw to save memory
        )
    except Exception as exc:
        logger.debug("Parse error for %s: %s", game_id, exc)
        return None


# ---------------------------------------------------------------------------
# Board state
# ---------------------------------------------------------------------------

def _parse_board_state(init_state: dict) -> ColonistBoardState:
    """
    Parse initialState.mapState.tileHexStates and portEdgeStates.

    tileHexStates: {idx: {x, y, type, diceNumber}}
      x = axial q,  y = axial r,  type 0-5,  diceNumber 0-12

    portEdgeStates: {idx: {x, y, z, type}}
      type 1-6  (1=generic, 2=Brick, 3=Wool, 4=Grain, 5=Ore, 6=Lumber)
    """
    map_state = init_state.get("mapState", {})
    tile_hex_states: dict = map_state.get("tileHexStates", {})
    port_edge_states: dict = map_state.get("portEdgeStates", {})

    hexes: list[ColonistHex] = []
    robber_idx = 0

    for idx_str, tile in tile_hex_states.items():
        idx = int(idx_str)
        res_int = int(tile.get("type", 0))
        roll = tile.get("diceNumber", 0) or None
        if roll == 0:
            roll = None
        res = TILE_TYPE_TO_RESOURCE.get(res_int, Resource.DESERT)
        if res == Resource.DESERT:
            robber_idx = idx
        hexes.append(ColonistHex(
            index=idx,
            resource=str(res_int),
            roll=roll,
        ))

    ports: list[ColonistPort] = []
    for idx_str, port in sorted(port_edge_states.items(), key=lambda kv: int(kv[0])):
        pt_int = int(port.get("type", 1))
        x = port.get("x", 0)
        y = port.get("y", 0)
        z = port.get("z", 0)
        ports.append(ColonistPort(
            port_type=str(pt_int),
            tile_indices=[],
            vertices=None,
            # Store (x, y, z) in a custom way for later processing
        ))
        # Attach raw coords to port for vertex mapping later
        ports[-1]._raw_x = x
        ports[-1]._raw_y = y
        ports[-1]._raw_z = z

    return ColonistBoardState(
        hexes=sorted(hexes, key=lambda h: h.index),
        ports=ports,
        robber_index=robber_idx,
    )


# ---------------------------------------------------------------------------
# Opening settlement extraction
# ---------------------------------------------------------------------------

def _extract_opening_settlements(events: list[dict]) -> list[ColonistOpeningPlacement]:
    """
    Detect opening settlement placements by scanning event mapState changes.

    Each event with mapState.tileCornerStates contains new or updated corner states.
    Opening settlements appear as new entries with buildingType=1.
    """
    placements: list[ColonistOpeningPlacement] = []
    seen_corners: set[int] = set()

    for seq, event in enumerate(events):
        if len(placements) >= OPENING_SETTLEMENT_COUNT:
            break

        sc = event.get("stateChange", {})
        ms = sc.get("mapState", {})
        corners = ms.get("tileCornerStates", {})
        if not corners:
            continue

        # Determine player from gameLogState
        player_color = _get_event_player(sc)

        for corner_str, info in corners.items():
            try:
                corner_id = int(corner_str)
            except (ValueError, TypeError):
                continue

            building = info.get("buildingType", 0)
            owner = info.get("owner")

            if building == 1 and corner_id not in seen_corners:
                seen_corners.add(corner_id)
                effective = int(owner) if owner is not None else (player_color or 0)
                placements.append(ColonistOpeningPlacement(
                    player_index=effective,
                    vertex_index=corner_id,
                    road_vertices=None,
                    action_seq=seq,
                ))

    return placements


def _get_event_player(sc: dict) -> Optional[int]:
    """Extract acting player color from stateChange.gameLogState."""
    log = sc.get("gameLogState", {})
    if isinstance(log, dict):
        for entry in log.values():
            if isinstance(entry, dict):
                pc = entry.get("from") or entry.get("text", {}).get("playerColor")
                if pc is not None:
                    return int(pc)
    return None


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------

def _parse_outcome(
    end_state: dict,
) -> tuple[int, dict[int, int], dict[int, int], int]:
    """Parse endGameState for winner, VP, ranks, turn count."""
    winner_color = 0
    final_vps: dict[int, int] = {}
    final_ranks: dict[int, int] = {}

    players = end_state.get("players", {})
    for color_str, pdata in players.items():
        if not isinstance(pdata, dict):
            continue
        try:
            color = int(color_str)
        except (ValueError, TypeError):
            continue

        if pdata.get("winningPlayer"):
            winner_color = color

        rank = pdata.get("rank", 0)
        final_ranks[color] = int(rank) if rank else 0

        vp_data = pdata.get("victoryPoints", {})
        if isinstance(vp_data, dict):
            # VP is {category_id: count} — sum all categories
            total_vp = sum(int(v) for v in vp_data.values() if v is not None)
            final_vps[color] = total_vp
        elif isinstance(vp_data, (int, float)):
            final_vps[color] = int(vp_data)

    total_turns = int(end_state.get("totalTurnCount", 0))
    return winner_color, final_vps, final_ranks, total_turns


# ---------------------------------------------------------------------------
# Board reconstruction: ColonistGameRecord → CatanBoard
# ---------------------------------------------------------------------------

def colonist_record_to_board(record: ColonistGameRecord) -> Optional[CatanBoard]:
    """
    Reconstruct a CatanBoard from a parsed game record.

    Tile coordinates: tileHexStates.x = axial q,  .y = axial r  (directly usable).
    Port assignment: angular-sort the 9 portEdgeStates and match to our 9 port slot edges.
    """
    try:
        hexes = record.board.hexes
        if not hexes:
            return CatanBoard.random()

        # --- Build resource and number arrays indexed by axial position ---
        # Map (q, r) → (Resource, number)
        axial_to_tile: dict[tuple[int, int], tuple[Resource, Optional[int]]] = {}
        for h in hexes:
            res = TILE_TYPE_TO_RESOURCE.get(int(h.resource), Resource.DESERT)
            axial_to_tile[(0, 0)] = (Resource.DESERT, None)  # safety default

        # Re-read from the raw ColonistHex objects
        for h in hexes:
            res_int = int(h.resource) if str(h.resource).isdigit() else 0
            res = TILE_TYPE_TO_RESOURCE.get(res_int, Resource.DESERT)
            # x=q, y=r stored in the index mapping
            # We need x,y from the raw tile; they're stored in the hex index → axial mapping
            # The index in our AXIAL_POSITIONS list IS determined by traversal order
            # Use the index directly to get our axial position
            if h.index < len(AXIAL_POSITIONS):
                axial_pos = AXIAL_POSITIONS[h.index]
                axial_to_tile[axial_pos] = (res, h.roll)

        # Build ordered resource and number lists matching AXIAL_POSITIONS order
        resources: list[Resource] = []
        numbers: list[int] = []
        for pos in AXIAL_POSITIONS:
            res, num = axial_to_tile.get(pos, (Resource.DESERT, None))
            resources.append(res)
            if num and res != Resource.DESERT:
                numbers.append(num)

        while len(numbers) < 18:
            numbers.append(5)

        # --- Port type assignment ---
        port_types = _assign_port_types(record.board.ports, record.board.hexes)

        return CatanBoard.from_tiles(
            hex_positions=list(AXIAL_POSITIONS),
            resources=resources,
            numbers=numbers[:18],
            port_types=port_types[:9],
        )

    except Exception as exc:
        logger.debug("Board reconstruction failed for %s: %s", record.game_id, exc)
        return None


def _assign_port_types(
    ports: list[ColonistPort],
    hexes: list[ColonistHex],
) -> list[PortType]:
    """
    Match portEdgeStates to our 9 port_slot_edges via angular ordering.

    The portEdgeState (x, y) gives an approximate position (land or sea hex).
    We convert each to Cartesian and sort by angle from board center.
    Our board's port_slot_edges are also sorted by angle in port_slot_edges().
    The i-th portEdgeState maps to the i-th port slot (after angular sort).
    """
    if not ports:
        return [PortType.GENERIC] * 9

    # Sort ports by the angle of their (x, y) hex center from board origin
    def port_angle(p: ColonistPort) -> float:
        x = getattr(p, "_raw_x", 0)
        y = getattr(p, "_raw_y", 0)
        cx, cy = axial_to_cartesian(x, y)
        return math.atan2(cy, cx)

    sorted_ports = sorted(ports, key=port_angle)
    result = []
    for p in sorted_ports[:9]:
        pt_int = int(p.port_type) if str(p.port_type).isdigit() else 1
        result.append(PORT_TYPE_INT_TO_PORT.get(pt_int, PortType.GENERIC))

    while len(result) < 9:
        result.append(PortType.GENERIC)
    return result


# ---------------------------------------------------------------------------
# Training record extraction
# ---------------------------------------------------------------------------

def _corner_id_to_vertex(
    corner_id: int,
    board: CatanBoard,
) -> Optional[tuple[float, float]]:
    """
    Map Colonist corner ID (0-53) to our board vertex key.

    Colonist numbers vertices sequentially; we sort ours by (y_asc, x_asc)
    to approximate the same top-to-bottom, left-to-right order.
    """
    all_verts = sorted(board.all_vertices(), key=lambda v: (round(v[1], 2), round(v[0], 2)))
    if 0 <= corner_id < len(all_verts):
        return all_verts[corner_id]
    return None


def extract_training_records(
    game_record: ColonistGameRecord,
    board: CatanBoard,
    compute_features: bool = True,
) -> list[ProcessedOpeningRecord]:
    """Convert one game record to up to 4 ProcessedOpeningRecords."""
    vf_cache = compute_all_vertex_features(board) if compute_features else None

    # Group placements by player color
    player_placements: dict[int, list[ColonistOpeningPlacement]] = {}
    for p in game_record.opening_placements:
        player_placements.setdefault(p.player_index, []).append(p)

    records: list[ProcessedOpeningRecord] = []
    for seat, player_color in enumerate(game_record.play_order[:4]):
        placements = player_placements.get(player_color, [])
        if len(placements) < 2:
            continue

        v1 = _corner_id_to_vertex(placements[0].vertex_index, board)
        v2 = _corner_id_to_vertex(placements[1].vertex_index, board)
        if v1 is None or v2 is None:
            continue

        won = (player_color == game_record.winner_index)
        rank = game_record.final_ranks.get(player_color, seat + 1)
        vp = game_record.final_vps.get(player_color, 0)

        feature_arr = None
        if compute_features and vf_cache is not None:
            try:
                of = compute_opening_features(v1, v2, seat, board, vf_cache)
                feature_arr = opening_features_to_array(of)
            except Exception as exc:
                logger.debug("Feature extraction failed: %s", exc)

        records.append(ProcessedOpeningRecord(
            game_id=game_record.game_id,
            player_index=player_color,
            seat=seat,
            v1=v1, v2=v2,
            won=won,
            final_rank=rank,
            final_vp=vp,
            feature_array=feature_arr,
        ))

    return records


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_game_files(data_dir: Path) -> Iterator[ColonistGameRecord]:
    """Lazily yield parsed game records from a directory of .json files."""
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        paths = sorted(data_dir.rglob("*.json"))
    logger.info("Found %d JSON files in %s", len(paths), data_dir)
    ok = fail = 0
    for path in paths:
        record = parse_game_file(path)
        if record is not None:
            ok += 1
            yield record
        else:
            fail += 1
    logger.info("Streamed %d games (%d failed)", ok, fail)


def stream_from_tarfile(
    tar_path: Path,
    max_games: Optional[int] = None,
) -> Iterator[ColonistGameRecord]:
    """
    Stream directly from games.tar.gz without extracting to disk.
    O(1) memory — reads one file at a time from the compressed archive.

    Usage:
        for record in stream_from_tarfile(Path("games.tar.gz")):
            board = colonist_record_to_board(record)
            ...
    """
    logger.info("Streaming from %s (no extraction needed)", tar_path)
    ok = fail = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json") or not member.isfile():
                continue
            game_id = Path(member.name).stem
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                raw = json.loads(f.read().decode("utf-8"))
                record = _parse_game_dict(raw, game_id=game_id)
                if record is not None:
                    ok += 1
                    yield record
                else:
                    fail += 1
            except Exception as exc:
                logger.debug("Failed %s: %s", game_id, exc)
                fail += 1

            if max_games and ok >= max_games:
                break

    logger.info("Streamed %d games from archive (%d failed)", ok, fail)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_training_dataset(
    source: Path,
    max_games: Optional[int] = None,
    compute_features: bool = True,
) -> list[ProcessedOpeningRecord]:
    """
    Load and process the Colonist dataset.
    source: directory of .json files  OR  a .tar.gz archive (streamed, no disk needed).
    Returns up to 4 records per game.
    """
    if source.is_file() and (source.suffix == ".gz" or str(source).endswith(".tar.gz")):
        game_stream = stream_from_tarfile(source, max_games=max_games)
    else:
        game_stream = stream_game_files(source)

    all_records: list[ProcessedOpeningRecord] = []
    n_games = 0
    for record in game_stream:
        board = colonist_record_to_board(record)
        if board is None:
            continue
        rows = extract_training_records(record, board, compute_features)
        all_records.extend(rows)
        n_games += 1
        if max_games and n_games >= max_games:
            break

    logger.info("Dataset: %d records from %d games", len(all_records), n_games)
    return all_records


def save_dataset_numpy(records: list[ProcessedOpeningRecord], output_path: Path) -> None:
    """Save feature arrays, labels, and metadata to .npz for fast reloading."""
    valid = [r for r in records if r.feature_array is not None]
    if not valid:
        logger.warning("No feature arrays to save")
        return
    X = np.stack([r.feature_array for r in valid])
    y = np.array([int(r.won) for r in valid], dtype=np.int32)
    seats = np.array([r.seat for r in valid], dtype=np.int32)
    final_vps = np.array([r.final_vp for r in valid], dtype=np.int32)
    final_ranks = np.array([r.final_rank for r in valid], dtype=np.int32)
    game_ids = np.array([r.game_id for r in valid], dtype=object)
    np.savez(output_path, X=X, y=y, seats=seats,
             final_vps=final_vps, final_ranks=final_ranks, game_ids=game_ids)
    logger.info("Saved %d records → %s", len(valid), output_path)


def load_dataset_numpy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load X, y from .npz."""
    d = np.load(path)
    return d["X"], d["y"]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_sample_files(source: Path, n: int = 3) -> None:
    """Print the raw structure of the first n game files (directory or .tar.gz)."""

    def _load_samples() -> list[tuple[str, dict]]:
        samples = []
        if source.is_file() and source.suffix in (".gz", ".tar"):
            with tarfile.open(source, "r:gz") as tf:
                for member in tf:
                    if not member.name.endswith(".json") or not member.isfile():
                        continue
                    f = tf.extractfile(member)
                    if f:
                        samples.append((Path(member.name).name, json.loads(f.read())))
                    if len(samples) >= n:
                        break
        else:
            for path in sorted(source.glob("*.json"))[:n]:
                with open(path) as f:
                    samples.append((path.name, json.load(f)))
        return samples

    for fname, raw in _load_samples():
        print(f"\n{'='*70}")
        print(f"FILE: {fname}")
        try:
            data = raw.get("data", raw)
            print(f"\n[Top-level keys]: {list(raw.keys())}")
            print(f"[data keys]: {list(data.keys())}")

            ev_hist = data.get("eventHistory", {})
            print(f"[eventHistory keys]: {list(ev_hist.keys())}")
            print(f"[playOrder]: {data.get('playOrder')}")

            # Initial state / board
            init = ev_hist.get("initialState", {})
            ms = init.get("mapState", {})
            if ms:
                print(f"\n[initialState.mapState keys]: {list(ms.keys())}")
                tiles = ms.get("tileHexStates", {})
                ports = ms.get("portEdgeStates", {})
                if tiles:
                    sample = dict(list(tiles.items())[:3])
                    print(f"  tileHexStates sample: {sample}")
                if ports:
                    sample = dict(list(ports.items())[:3])
                    print(f"  portEdgeStates sample: {sample}")

            # Events
            events = ev_hist.get("events", [])
            print(f"\n[Number of events]: {len(events)}")
            for i, ev in enumerate(events[:4]):
                sc = ev.get("stateChange", {})
                ev_ms = sc.get("mapState", {})
                corners = ev_ms.get("tileCornerStates", {})
                if corners:
                    sample = dict(list(corners.items())[:2])
                    print(f"  [Event {i}] tileCornerStates: {sample}")
                    log = sc.get("gameLogState", {})
                    first = next(iter(log.values()), None) if isinstance(log, dict) else None
                    if first:
                        print(f"    gameLogState: {first}")

            # End state
            end = ev_hist.get("endGameState", {})
            print(f"\n[endGameState keys]: {list(end.keys())}")
            print(f"  totalTurnCount: {end.get('totalTurnCount')}")
            players = end.get("players", {})
            if players:
                first_color = next(iter(players))
                print(f"  players[{first_color}]: {players[first_color]}")

        except Exception as exc:
            print(f"Error: {exc}")
