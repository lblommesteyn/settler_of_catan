"""
CatanBoard: full board representation combining hex tiles, vertex graph, and ports.

Immutable once constructed — all mutable game state lives in simulation/game_state.py.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hex_grid import BoardGraph, AXIAL_POSITIONS, axial_to_cartesian


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


class PortType(Enum):
    GENERIC = "3:1"
    WOOD = "2:1_wood"
    BRICK = "2:1_brick"
    SHEEP = "2:1_sheep"
    WHEAT = "2:1_wheat"
    ORE = "2:1_ore"


# Pip values (number of ways to roll on 2d6)
PIP_VALUES: dict[int, int] = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}

# Standard distribution of resource tiles (19 total)
RESOURCE_POOL: list[Resource] = (
    [Resource.WOOD] * 4
    + [Resource.BRICK] * 3
    + [Resource.SHEEP] * 4
    + [Resource.WHEAT] * 4
    + [Resource.ORE] * 3
    + [Resource.DESERT] * 1
)

# Standard number tokens (18 for 18 non-desert hexes)
NUMBER_POOL: list[int] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# Standard port types (9 ports)
PORT_POOL: list[PortType] = [
    PortType.GENERIC, PortType.GENERIC, PortType.GENERIC, PortType.GENERIC,
    PortType.WOOD, PortType.BRICK, PortType.SHEEP, PortType.WHEAT, PortType.ORE,
]

# Port trade ratios
PORT_RATIO: dict[PortType, float] = {
    PortType.GENERIC: 1 / 3,
    PortType.WOOD: 1 / 2,
    PortType.BRICK: 1 / 2,
    PortType.SHEEP: 1 / 2,
    PortType.WHEAT: 1 / 2,
    PortType.ORE: 1 / 2,
}

PORT_RESOURCE: dict[PortType, Optional[Resource]] = {
    PortType.GENERIC: None,
    PortType.WOOD: Resource.WOOD,
    PortType.BRICK: Resource.BRICK,
    PortType.SHEEP: Resource.SHEEP,
    PortType.WHEAT: Resource.WHEAT,
    PortType.ORE: Resource.ORE,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HexTile:
    q: int
    r: int
    resource: Resource
    number: Optional[int]  # None for desert

    @property
    def pips(self) -> int:
        return PIP_VALUES.get(self.number, 0) if self.number else 0

    @property
    def cartesian(self) -> tuple[float, float]:
        return axial_to_cartesian(self.q, self.r)


@dataclass(frozen=True)
class Port:
    port_type: PortType
    vertices: frozenset  # frozenset of 2 vertex_key tuples

    @property
    def ratio(self) -> float:
        return PORT_RATIO[self.port_type]

    @property
    def resource(self) -> Optional[Resource]:
        return PORT_RESOURCE[self.port_type]


# ---------------------------------------------------------------------------
# CatanBoard
# ---------------------------------------------------------------------------

@dataclass
class CatanBoard:
    """
    Immutable Catan board. Stores hex tiles, the vertex/edge graph, and ports.

    Key attributes:
      graph       — BoardGraph with vertex adjacency
      tiles       — {(q,r): HexTile}
      vertex_hexes — {vertex_key: [HexTile, ...]}  (1–3 tiles per vertex)
      vertex_port  — {vertex_key: Port | None}
      robber_start — (q, r) of desert hex
    """
    graph: BoardGraph
    tiles: dict[tuple[int, int], HexTile]
    vertex_hexes: dict[tuple[float, float], list[HexTile]]
    ports: list[Port]
    vertex_port: dict[tuple[float, float], Optional[Port]]
    robber_start: tuple[int, int]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "CatanBoard":
        """Generate a fully randomised standard 4-player Catan board."""
        rng = random.Random(seed)

        resources = RESOURCE_POOL[:]
        rng.shuffle(resources)

        numbers = NUMBER_POOL[:]
        rng.shuffle(numbers)

        port_types = PORT_POOL[:]
        rng.shuffle(port_types)

        return cls._build(AXIAL_POSITIONS, resources, numbers, port_types)

    @classmethod
    def from_tiles(
        cls,
        hex_positions: list[tuple[int, int]],
        resources: list[Resource],
        numbers: list[int],
        port_types: list[PortType],
    ) -> "CatanBoard":
        """Build a board from explicit tile and port assignments."""
        return cls._build(hex_positions, resources, numbers, port_types)

    @classmethod
    def _build(
        cls,
        hex_positions: list[tuple[int, int]],
        resources: list[Resource],
        numbers: list[int],
        port_types: list[PortType],
    ) -> "CatanBoard":
        graph = BoardGraph(hex_positions)

        # Assign resources and numbers to hexes
        tiles: dict[tuple[int, int], HexTile] = {}
        number_iter = iter(numbers)
        robber_start = (0, 0)

        for (q, r), res in zip(hex_positions, resources):
            num = None if res == Resource.DESERT else next(number_iter)
            tile = HexTile(q=q, r=r, resource=res, number=num)
            tiles[(q, r)] = tile
            if res == Resource.DESERT:
                robber_start = (q, r)

        # Build vertex_hexes lookup
        vertex_hexes: dict[tuple[float, float], list[HexTile]] = {}
        for vk, hex_coords in graph.vertices.items():
            vertex_hexes[vk] = [tiles[hc] for hc in hex_coords if hc in tiles]

        # Assign ports to coastal edge slots (9 evenly-spaced perimeter edges)
        coastal_sorted = graph.port_slot_edges(n_ports=len(port_types))
        ports: list[Port] = []
        vertex_port: dict[tuple[float, float], Optional[Port]] = {
            vk: None for vk in graph.vertices
        }
        for i, (ek, pt) in enumerate(zip(coastal_sorted, port_types)):
            v1, v2 = tuple(ek)
            port = Port(port_type=pt, vertices=frozenset({v1, v2}))
            ports.append(port)
            vertex_port[v1] = port
            vertex_port[v2] = port

        return cls(
            graph=graph,
            tiles=tiles,
            vertex_hexes=vertex_hexes,
            ports=ports,
            vertex_port=vertex_port,
            robber_start=robber_start,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_vertices(self) -> list[tuple[float, float]]:
        return self.graph.all_vertices()

    def pip_count(self, vk: tuple[float, float]) -> int:
        """Total pip dots for all number tokens adjacent to this vertex."""
        return sum(t.pips for t in self.vertex_hexes.get(vk, []))

    def resources_at(self, vk: tuple[float, float]) -> set[Resource]:
        """Set of (non-desert) resources adjacent to this vertex."""
        return {
            t.resource
            for t in self.vertex_hexes.get(vk, [])
            if t.resource != Resource.DESERT
        }

    def get_port(self, vk: tuple[float, float]) -> Optional[Port]:
        return self.vertex_port.get(vk)

    def legal_starting_vertices(self) -> list[tuple[float, float]]:
        """All 54 vertices are legal for the very first placement."""
        return [vk for vk in self.graph.vertices if self.vertex_hexes.get(vk)]

    def legal_second_vertices(
        self,
        first: tuple[float, float],
        occupied: Optional[set[tuple[float, float]]] = None,
    ) -> list[tuple[float, float]]:
        """
        Legal vertices for a second settlement given one already placed.
        Distance rule: must be at least 2 road steps from any existing settlement.
        """
        if occupied is None:
            occupied = {first}
        else:
            occupied = occupied | {first}

        # BFS from all occupied vertices; any vertex within distance 1 is illegal
        too_close: set[tuple[float, float]] = set()
        for occ in occupied:
            too_close.add(occ)
            too_close.update(self.graph.vertex_neighbors[occ])

        return [
            vk
            for vk in self.legal_starting_vertices()
            if vk not in too_close
        ]

    def snake_draft_order(self, n_players: int = 4) -> list[int]:
        """
        Returns seat indices in snake-draft order for opening placements.
        e.g. 4 players: [0, 1, 2, 3, 3, 2, 1, 0]
        """
        forward = list(range(n_players))
        return forward + list(reversed(forward))
