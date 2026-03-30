"""
Data schema for parsed Colonist.io game records.

The Colonist dataset format is documented by inspection of sample files.
Fields may vary; all parsers should fail gracefully on missing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ColonistHex:
    """One land hex from initialState."""
    index: int             # Colonist tile index (0-18)
    resource: str          # "FOREST", "HILLS", "PASTURE", "FIELDS", "MOUNTAINS", "DESERT"
    roll: Optional[int]    # number token (None for desert)


@dataclass
class ColonistPort:
    """One port from initialState."""
    port_type: str         # "3_1", "2_1_WOOD", etc. (varies by dataset version)
    tile_indices: list[int]  # which tile indices are adjacent
    vertices: Optional[list[int]]  # vertex indices if directly encoded


@dataclass
class ColonistBoardState:
    hexes: list[ColonistHex]
    ports: list[ColonistPort]
    robber_index: int  # index of desert hex


@dataclass
class ColonistOpeningPlacement:
    player_index: int
    vertex_index: int      # Colonist vertex ID
    road_vertices: Optional[tuple[int, int]]  # the two endpoints of the road
    action_seq: int        # position in the event stream


@dataclass
class ColonistGameRecord:
    game_id: str
    board: ColonistBoardState
    play_order: list[int]            # player indices in seat order
    opening_placements: list[ColonistOpeningPlacement]  # should be 8 (4 settlements + 4 roads)
    winner_index: int
    final_vps: dict[int, int]        # {player_index: vp}
    final_ranks: dict[int, int]      # {player_index: rank 1-4}
    total_turns: int
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class ProcessedOpeningRecord:
    """One training row. Each game produces up to 4 rows (one per player)."""
    game_id: str
    player_index: int
    seat: int              # 0-3 (position in play_order)
    # Opening vertices (as Cartesian keys after board mapping)
    v1: tuple[float, float]
    v2: tuple[float, float]
    # Labels
    won: bool
    final_rank: int        # 1 = best
    final_vp: int
    # Features (filled by feature pipeline)
    feature_array: Optional[Any] = None  # np.ndarray shape (75,)
