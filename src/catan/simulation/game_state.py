"""
Mutable game state for a Catan simulation.

The board itself is immutable (CatanBoard); this class tracks
all the mutable per-game state: resources, buildings, VP, robber position.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from ..board.board import CatanBoard, Resource, PIP_VALUES


ResourceHand = dict[Resource, int]

SETTLEMENT_COST: ResourceHand = {
    Resource.WOOD: 1, Resource.BRICK: 1, Resource.SHEEP: 1, Resource.WHEAT: 1,
}
ROAD_COST: ResourceHand = {Resource.WOOD: 1, Resource.BRICK: 1}
CITY_COST: ResourceHand = {Resource.WHEAT: 2, Resource.ORE: 3}
DEV_COST: ResourceHand = {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1}

VP_FOR_SETTLEMENT = 1
VP_FOR_CITY = 2
VP_FOR_LONGEST_ROAD = 2
VP_FOR_LARGEST_ARMY = 2
LONGEST_ROAD_THRESHOLD = 5
LARGEST_ARMY_THRESHOLD = 3


def empty_hand() -> ResourceHand:
    return {r: 0 for r in Resource if r != Resource.DESERT}


@dataclass
class PlayerState:
    player_id: int
    seat: int
    resources: ResourceHand = field(default_factory=empty_hand)
    settlements: list[tuple[float, float]] = field(default_factory=list)
    cities: list[tuple[float, float]] = field(default_factory=list)
    roads: list[frozenset] = field(default_factory=list)
    knights_played: int = 0
    vp_from_dev_cards: int = 0  # hidden VP cards

    # Tracks when first city / first expansion were achieved (turn number)
    first_city_turn: Optional[int] = None
    first_expansion_turn: Optional[int] = None  # 3rd settlement

    def total_vp(self, has_longest_road: bool = False, has_largest_army: bool = False) -> int:
        vp = (
            len(self.settlements) * VP_FOR_SETTLEMENT
            + len(self.cities) * VP_FOR_CITY
            + self.vp_from_dev_cards
        )
        if has_longest_road:
            vp += VP_FOR_LONGEST_ROAD
        if has_largest_army:
            vp += VP_FOR_LARGEST_ARMY
        return vp

    def can_afford(self, cost: ResourceHand) -> bool:
        return all(self.resources.get(r, 0) >= n for r, n in cost.items() if n > 0)

    def spend(self, cost: ResourceHand) -> None:
        for r, n in cost.items():
            if n > 0:
                self.resources[r] -= n

    def gain(self, gains: ResourceHand) -> None:
        for r, n in gains.items():
            self.resources[r] = self.resources.get(r, 0) + n

    def hand_size(self) -> int:
        return sum(self.resources.values())


@dataclass
class GameState:
    board: CatanBoard
    players: list[PlayerState]
    robber_hex: tuple[int, int]
    current_turn: int = 0
    phase: str = "setup"  # "setup" | "main" | "done"
    winner_id: Optional[int] = None

    # Shared special cards state
    longest_road_holder: Optional[int] = None
    longest_road_length: int = 0
    largest_army_holder: Optional[int] = None
    largest_army_size: int = 0

    # Occupied vertices and edges (for legality checks)
    occupied_vertices: dict[tuple[float, float], int] = field(default_factory=dict)  # vertex -> player_id
    occupied_edges: dict[frozenset, int] = field(default_factory=dict)  # edge -> player_id

    @classmethod
    def new_game(cls, board: CatanBoard, n_players: int = 4) -> "GameState":
        players = [
            PlayerState(player_id=i, seat=i) for i in range(n_players)
        ]
        return cls(
            board=board,
            players=players,
            robber_hex=board.robber_start,
        )

    # ------------------------------------------------------------------
    # Setup-phase actions (free placements)
    # ------------------------------------------------------------------

    def place_setup_settlement(
        self, player_id: int, vk: tuple[float, float], receive_resources: bool = False
    ) -> None:
        """Place a free setup settlement. receive_resources=True for second settlement."""
        p = self.players[player_id]
        p.settlements.append(vk)
        self.occupied_vertices[vk] = player_id

        if receive_resources:
            for tile in self.board.vertex_hexes.get(vk, []):
                if tile.resource != Resource.DESERT and tile.number is not None:
                    p.resources[tile.resource] = p.resources.get(tile.resource, 0) + 1

    def place_setup_road(
        self, player_id: int, edge: frozenset
    ) -> None:
        """Place a free setup road."""
        p = self.players[player_id]
        p.roads.append(edge)
        self.occupied_edges[edge] = player_id

    # ------------------------------------------------------------------
    # Main-game actions
    # ------------------------------------------------------------------

    def distribute_resources(self, roll: int) -> dict[int, ResourceHand]:
        """Distribute resources for a dice roll. Returns {player_id: gained}."""
        gains: dict[int, ResourceHand] = {p.player_id: empty_hand() for p in self.players}

        if roll == 7:
            return gains  # robber logic handled separately

        for vk, pid in self.occupied_vertices.items():
            multiplier = 2 if vk in self.players[pid].cities else 1
            for tile in self.board.vertex_hexes.get(vk, []):
                if tile.number == roll and tile.resource != Resource.DESERT:
                    gains[pid][tile.resource] = gains[pid].get(tile.resource, 0) + multiplier

        for pid, gained in gains.items():
            self.players[pid].gain(gained)

        return gains

    def build_road(self, player_id: int, edge: frozenset, free: bool = False) -> bool:
        """Build a road. Returns True if successful."""
        p = self.players[player_id]
        if edge in self.occupied_edges:
            return False
        # Must be adjacent to own settlement, city, or existing road
        v1, v2 = tuple(edge)
        own_vertices = set(p.settlements) | set(p.cities)
        connected = (
            v1 in own_vertices or v2 in own_vertices
            or any(
                v1 in frozenset(r) or v2 in frozenset(r)
                for r in p.roads
            )
        )
        if not connected:
            return False
        if not free and not p.can_afford(ROAD_COST):
            return False
        if not free:
            p.spend(ROAD_COST)
        p.roads.append(edge)
        self.occupied_edges[edge] = player_id
        self._update_longest_road()
        return True

    def build_settlement(self, player_id: int, vk: tuple[float, float], free: bool = False) -> bool:
        """Build a settlement. Returns True if successful."""
        p = self.players[player_id]
        if vk in self.occupied_vertices:
            return False
        # Distance rule
        for nb in self.board.graph.vertex_neighbors[vk]:
            if nb in self.occupied_vertices:
                return False
        # Must be connected by own road (except setup)
        road_vertices = {v for r in p.roads for v in tuple(r)}
        if not free and vk not in road_vertices:
            return False
        if not free and not p.can_afford(SETTLEMENT_COST):
            return False
        if not free:
            p.spend(SETTLEMENT_COST)
        p.settlements.append(vk)
        self.occupied_vertices[vk] = player_id

        if p.first_expansion_turn is None and len(p.settlements) >= 3:
            p.first_expansion_turn = self.current_turn
        return True

    def build_city(self, player_id: int, vk: tuple[float, float]) -> bool:
        """Upgrade a settlement to a city."""
        p = self.players[player_id]
        if vk not in p.settlements:
            return False
        if not p.can_afford(CITY_COST):
            return False
        p.spend(CITY_COST)
        p.settlements.remove(vk)
        p.cities.append(vk)
        if p.first_city_turn is None:
            p.first_city_turn = self.current_turn
        return True

    def move_robber(self, player_id: int, target_hex: tuple[int, int]) -> Optional[int]:
        """Move robber to target hex. Returns player_id of a stolen resource (or None)."""
        self.robber_hex = target_hex
        # Steal one random resource from a player on that hex
        victims = set()
        for vk, pid in self.occupied_vertices.items():
            if pid != player_id:
                for tile in self.board.vertex_hexes.get(vk, []):
                    if (tile.q, tile.r) == target_hex:
                        victims.add(pid)
        return victims.pop() if victims else None

    def check_winner(self) -> Optional[int]:
        """Return player_id if someone has >= 10 VP, else None."""
        for p in self.players:
            vp = p.total_vp(
                has_longest_road=(self.longest_road_holder == p.player_id),
                has_largest_army=(self.largest_army_holder == p.player_id),
            )
            if vp >= 10:
                self.winner_id = p.player_id
                self.phase = "done"
                return p.player_id
        return None

    def vp_for_player(self, player_id: int) -> int:
        p = self.players[player_id]
        return p.total_vp(
            has_longest_road=(self.longest_road_holder == player_id),
            has_largest_army=(self.largest_army_holder == player_id),
        )

    def _update_longest_road(self) -> None:
        """Simplified longest road: count roads per player (not true longest path)."""
        for p in self.players:
            road_len = len(p.roads)
            if road_len > self.longest_road_length and road_len >= LONGEST_ROAD_THRESHOLD:
                self.longest_road_length = road_len
                self.longest_road_holder = p.player_id

    def _update_largest_army(self) -> None:
        for p in self.players:
            if (
                p.knights_played > self.largest_army_size
                and p.knights_played >= LARGEST_ARMY_THRESHOLD
            ):
                self.largest_army_size = p.knights_played
                self.largest_army_holder = p.player_id

    def clone(self) -> "GameState":
        return copy.deepcopy(self)
