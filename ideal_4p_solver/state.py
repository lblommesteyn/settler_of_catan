"""State and rules interfaces for an ideal 4-player Catan solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


VertexId = int
EdgeId = int
HexId = int
PlayerId = int
PayoffVector = tuple[float, float, float, float]


class Resource(str, Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


class DevCardType(str, Enum):
    KNIGHT = "knight"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"
    VICTORY_POINT = "victory_point"


class TurnPhase(str, Enum):
    PRE_ROLL = "pre_roll"
    RESOLVE_SEVEN = "resolve_seven"
    MAIN = "main"
    PENDING_TRADE = "pending_trade"
    GAME_OVER = "game_over"


ResourceCounts = dict[Resource, int]


@dataclass(frozen=True)
class HexTile:
    hex_id: HexId
    resource: Optional[Resource]
    number: Optional[int]


@dataclass(frozen=True)
class Port:
    edge_id: EdgeId
    ratio: int
    resource: Optional[Resource]


@dataclass(frozen=True)
class BoardSnapshot:
    """Immutable board layout used by the full solver."""

    hexes: tuple[HexTile, ...]
    ports: tuple[Port, ...]
    vertex_count: int
    edge_count: int


@dataclass(frozen=True)
class PublicPlayerState:
    player_id: PlayerId
    visible_vp: int
    settlements: frozenset[VertexId]
    cities: frozenset[VertexId]
    roads: frozenset[EdgeId]
    hand_size: int
    dev_cards_bought: int
    knights_played: int
    longest_road_length: int
    harbors: frozenset[int]
    settlements_left: int
    cities_left: int
    roads_left: int


@dataclass(frozen=True)
class PrivatePlayerState:
    player_id: PlayerId
    resources: ResourceCounts
    dev_cards_in_hand: dict[DevCardType, int]
    hidden_vp_cards: int


@dataclass(frozen=True)
class TradeOffer:
    offerer: PlayerId
    responder: Optional[PlayerId]
    give: ResourceCounts
    receive: ResourceCounts


@dataclass(frozen=True)
class PendingTrade:
    offer: TradeOffer
    responders_left: tuple[PlayerId, ...]


@dataclass(frozen=True)
class PublicGameState:
    board: BoardSnapshot
    players: tuple[PublicPlayerState, ...]
    current_player: PlayerId
    phase: TurnPhase
    robber_hex: HexId
    bank_resources: ResourceCounts
    dev_deck_size: int
    dev_cards_played_this_turn: int
    dice_rolled_this_turn: bool
    pending_trade: Optional[PendingTrade]
    turn_number: int
    winner: Optional[PlayerId]


@dataclass(frozen=True)
class FullGameState:
    """
    Full engine state.

    Search should not expose this directly to the acting policy except through
    `Observation`; hidden information belongs in the engine or belief tracker.
    """

    public: PublicGameState
    private_players: tuple[PrivatePlayerState, ...]
    dev_deck_order: tuple[DevCardType, ...]
    pending_discarders: tuple[PlayerId, ...] = field(default_factory=tuple)
    payoff_vector: Optional[PayoffVector] = None


@dataclass(frozen=True)
class Observation:
    """What a player may condition on at decision time."""

    public: PublicGameState
    self_private: PrivatePlayerState
    actor_id: PlayerId


class RulesEngine(Protocol):
    """Exact-rules engine required by search, self-play, and evaluation."""

    def initial_state(self, board: BoardSnapshot, seed: Optional[int] = None) -> FullGameState:
        """Return a fully initialized hidden-information game state."""

    def observation_for(self, state: FullGameState, player_id: PlayerId) -> Observation:
        """Project the full state into the acting player's information set."""

    def legal_actions(self, state: FullGameState) -> list["Action"]:
        """Enumerate all legal atomic actions for the current phase."""

    def apply_action(
        self,
        state: FullGameState,
        action: "Action",
        seed: Optional[int] = None,
    ) -> FullGameState:
        """Advance the game by one legal action, including chance outcomes."""

    def is_terminal(self, state: FullGameState) -> bool:
        """Return whether the game is over."""

    def terminal_payoffs(self, state: FullGameState) -> PayoffVector:
        """
        Return a 4-player payoff vector.

        The simplest convention is win utility: 1.0 for the winner, 0.0 for
        everyone else. More shaped utilities can be used for training, but the
        engine should expose the exact terminal payoff separately.
        """

