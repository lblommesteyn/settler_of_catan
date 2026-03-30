"""
Hex grid math for Catan board construction.

Uses pointy-top axial coordinates (q, r).
- Center of hex (q, r): x = sqrt(3)*q + sqrt(3)/2*r,  y = 3/2*r
- 6 corners at angles 30°+60°*i from center
- Vertices identified by rounded Cartesian position (stable dict keys)
- Edges identified by frozenset of 2 vertex keys
"""

import math
from collections import defaultdict
from typing import Iterator

SQRT3 = math.sqrt(3)
COORD_PRECISION = 6  # decimal places for vertex key rounding

# Standard 4-player Catan board: 19 hexes in a hexagonal arrangement.
# Axial (q, r) positions: ring 0 + ring 1 + ring 2
AXIAL_POSITIONS: list[tuple[int, int]] = [
    # Ring 0 — center
    (0, 0),
    # Ring 1 — 6 hexes
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
    # Ring 2 — 12 hexes
    (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
    (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1),
]

# 6 axial neighbor directions (pointy-top)
AXIAL_DIRECTIONS: list[tuple[int, int]] = [
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
]

# Corner angles for pointy-top hexes (degrees, counterclockwise from 30°)
CORNER_ANGLES_DEG = [30 + 60 * i for i in range(6)]


def axial_to_cartesian(q: int, r: int) -> tuple[float, float]:
    """Pointy-top axial to Cartesian, unit hex size."""
    x = SQRT3 * q + SQRT3 / 2 * r
    y = 3 / 2 * r
    return (x, y)


def vertex_key(x: float, y: float) -> tuple[float, float]:
    """Round Cartesian coords to stable dict key."""
    return (round(x, COORD_PRECISION), round(y, COORD_PRECISION))


def hex_corners(q: int, r: int) -> list[tuple[float, float]]:
    """Return 6 Cartesian corner positions of the hex at axial (q, r)."""
    cx, cy = axial_to_cartesian(q, r)
    corners = []
    for deg in CORNER_ANGLES_DEG:
        rad = math.radians(deg)
        vx = cx + math.cos(rad)
        vy = cy + math.sin(rad)
        corners.append(vertex_key(vx, vy))
    return corners


def axial_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex grid distance between two axial positions."""
    dq, dr = q2 - q1, r2 - r1
    return max(abs(dq), abs(dr), abs(dq + dr))


def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
    """Return 6 axial neighbors (may include positions outside the board)."""
    return [(q + dq, r + dr) for dq, dr in AXIAL_DIRECTIONS]


class BoardGraph:
    """
    Encodes the full vertex and edge graph of the 19-hex Catan board.

    After construction:
      - self.vertices: dict mapping vertex_key -> list of adjacent hex (q,r)
      - self.edges: dict mapping frozenset{v1_key, v2_key} -> (v1_key, v2_key)
      - self.vertex_neighbors: dict mapping vertex_key -> list of adjacent vertex_keys
      - self.hex_vertices: dict mapping (q,r) -> list of 6 vertex_keys
      - self.hex_set: set of all valid (q,r) positions
    """

    def __init__(self, hex_positions: list[tuple[int, int]] = AXIAL_POSITIONS):
        self.hex_set: set[tuple[int, int]] = set(hex_positions)
        self.hex_vertices: dict[tuple[int, int], list[tuple[float, float]]] = {}
        self.vertices: dict[tuple[float, float], list[tuple[int, int]]] = defaultdict(list)
        self.edges: dict[frozenset, tuple[tuple[float, float], tuple[float, float]]] = {}
        self.vertex_neighbors: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
        self._dist_cache: dict[tuple[float, float], dict[tuple[float, float], int]] = {}
        self._build()

    def _build(self) -> None:
        # Step 1: Compute corners for every hex and collect unique vertices
        for q, r in self.hex_set:
            corners = hex_corners(q, r)
            self.hex_vertices[(q, r)] = corners
            for vk in corners:
                if (q, r) not in self.vertices[vk]:
                    self.vertices[vk].append((q, r))

        # Step 2: Build edges (consecutive corner pairs of each hex)
        for q, r in self.hex_set:
            corners = self.hex_vertices[(q, r)]
            n = len(corners)
            for i in range(n):
                v1 = corners[i]
                v2 = corners[(i + 1) % n]
                ek = frozenset({v1, v2})
                if ek not in self.edges:
                    self.edges[ek] = (v1, v2)

        # Step 3: Build vertex adjacency from edges
        for v1, v2 in self.edges.values():
            if v2 not in self.vertex_neighbors[v1]:
                self.vertex_neighbors[v1].append(v2)
            if v1 not in self.vertex_neighbors[v2]:
                self.vertex_neighbors[v2].append(v1)

    def all_vertices(self) -> list[tuple[float, float]]:
        return list(self.vertices.keys())

    def is_coastal_vertex(self, vk: tuple[float, float]) -> bool:
        """A coastal vertex touches fewer than 3 land hexes."""
        return len(self.vertices[vk]) < 3

    def coastal_vertices(self) -> list[tuple[float, float]]:
        return [v for v in self.vertices if self.is_coastal_vertex(v)]

    def coastal_edges(self) -> list[frozenset]:
        """All perimeter edges (both vertices coastal, sharing exactly 1 hex).
        Standard 19-hex board has 30 such edges.
        """
        result = []
        for ek, (v1, v2) in self.edges.items():
            if self.is_coastal_vertex(v1) and self.is_coastal_vertex(v2):
                shared = set(self.vertices[v1]) & set(self.vertices[v2])
                if len(shared) == 1:
                    result.append(ek)
        return result

    def port_slot_edges(self, n_ports: int = 9) -> list[frozenset]:
        """
        Select n_ports evenly-spaced coastal edges for port placement.
        Sorts all coastal edges by the angle of their midpoint from the
        board center, then picks n_ports evenly distributed around the ring.
        """
        import math
        coastal = self.coastal_edges()
        if not coastal:
            return []

        def edge_angle(ek: frozenset) -> float:
            v1, v2 = tuple(ek)
            mx = (v1[0] + v2[0]) / 2
            my = (v1[1] + v2[1]) / 2
            return math.atan2(my, mx)

        sorted_edges = sorted(coastal, key=edge_angle)
        total = len(sorted_edges)
        # Pick n_ports evenly spaced
        step = total / n_ports
        indices = [round(step * i) % total for i in range(n_ports)]
        return [sorted_edges[i] for i in sorted(set(indices))]

    def vertex_road_distance(
        self,
        source: tuple[float, float],
        max_depth: int = 99,
    ) -> dict[tuple[float, float], int]:
        """BFS from source vertex. Returns {vertex: road_steps}. Results are cached."""
        if source in self._dist_cache:
            return self._dist_cache[source]
        dist: dict[tuple[float, float], int] = {source: 0}
        queue = [source]
        while queue:
            next_queue = []
            for v in queue:
                d = dist[v]
                if d >= max_depth:
                    continue
                for nb in self.vertex_neighbors[v]:
                    if nb not in dist:
                        dist[nb] = d + 1
                        next_queue.append(nb)
            queue = next_queue
        self._dist_cache[source] = dist
        return dist

    def reachable_vertices(
        self,
        sources: list[tuple[float, float]],
        max_steps: int,
        exclude: set[tuple[float, float]] | None = None,
    ) -> set[tuple[float, float]]:
        """BFS from multiple sources up to max_steps. Optionally exclude vertices."""
        visited: dict[tuple[float, float], int] = {}
        for s in sources:
            visited[s] = 0
        queue = list(sources)
        while queue:
            next_queue = []
            for v in queue:
                d = visited[v]
                if d >= max_steps:
                    continue
                for nb in self.vertex_neighbors[v]:
                    if nb not in visited:
                        visited[nb] = d + 1
                        next_queue.append(nb)
            queue = next_queue
        result = set(visited.keys()) - set(sources)
        if exclude:
            result -= exclude
        return result

    def distance_between(
        self, v1: tuple[float, float], v2: tuple[float, float]
    ) -> int:
        """Road-step distance between two vertices (BFS)."""
        dist = self.vertex_road_distance(v1)
        return dist.get(v2, 999)
