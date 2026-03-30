"""Exact-state schema for a full 4-player Catan solver foundation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..board.board import CatanBoard, PortType, Resource


ResourceHand = dict[Resource, int]


def playable_resources() -> tuple[Resource, ...]:
    """All resource-card types in the base game."""
    return tuple(r for r in Resource if r != Resource.DESERT)


def empty_hand() -> ResourceHand:
    """Return a zeroed resource hand for the five base resources."""
    return {resource: 0 for resource in playable_resources()}


def normalize_hand(hand: Optional[ResourceHand] = None) -> ResourceHand:
    """Return a canonical five-resource hand with non-negative integer counts."""
    normalized = empty_hand()
    if hand is None:
        return normalized

    for resource, amount in hand.items():
        if resource == Resource.DESERT:
            continue
        normalized[resource] = int(amount)

    for resource, amount in normalized.items():
        if amount < 0:
            raise ValueError(f"Negative resource count for {resource.value}: {amount}")
    return normalized


class DevCardType(str, Enum):
    KNIGHT = "knight"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"
    VICTORY_POINT = "victory_point"


class TurnPhase(str, Enum):
    SETUP = "setup"
    PRE_ROLL = "pre_roll"
    MAIN = "main"
    PENDING_TRADE = "pending_trade"
    RESOLVE_SEVEN = "resolve_seven"
    GAME_OVER = "game_over"


DEV_CARD_DISTRIBUTION: dict[DevCardType, int] = {
    DevCardType.KNIGHT: 14,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
    DevCardType.VICTORY_POINT: 5,
}

RESOURCE_BANK_START = 19
LONGEST_ROAD_MIN_LENGTH = 5
LARGEST_ARMY_MIN_KNIGHTS = 3
ROAD_COST: ResourceHand = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
}
SETTLEMENT_COST: ResourceHand = {
    Resource.WOOD: 1,
    Resource.BRICK: 1,
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
}
CITY_COST: ResourceHand = {
    Resource.WHEAT: 2,
    Resource.ORE: 3,
}
DEV_CARD_COST: ResourceHand = {
    Resource.SHEEP: 1,
    Resource.WHEAT: 1,
    Resource.ORE: 1,
}


@dataclass(frozen=True)
class TradeOffer:
    """
    Domestic trade from the current player to one opponent or the table.

    `give` and `receive` are from the offerer's perspective.
    """

    offerer: int
    responder: Optional[int]
    give: ResourceHand
    receive: ResourceHand

    def canonical(self) -> "TradeOffer":
        """Return the offer with normalized resource hands."""
        return TradeOffer(
            offerer=self.offerer,
            responder=self.responder,
            give=normalize_hand(self.give),
            receive=normalize_hand(self.receive),
        )


@dataclass(frozen=True)
class PendingTrade:
    """A domestic offer awaiting responses."""

    offer: TradeOffer
    responders_left: tuple[int, ...]


@dataclass(frozen=True)
class MaritimeTrade:
    """Bank/harbor trade executed by the current player."""

    player_id: int
    give_resource: Resource
    give_count: int
    receive_resource: Resource


@dataclass(frozen=True)
class PublicPlayerState:
    player_id: int
    settlements: frozenset[tuple[float, float]] = field(default_factory=frozenset)
    cities: frozenset[tuple[float, float]] = field(default_factory=frozenset)
    roads: frozenset[frozenset] = field(default_factory=frozenset)
    hand_size: int = 0
    visible_vp: int = 0
    played_knights: int = 0
    dev_cards_bought: int = 0
    longest_road_length: int = 0
    has_longest_road: bool = False
    has_largest_army: bool = False
    ports: tuple[PortType, ...] = field(default_factory=tuple)
    settlements_left: int = 5
    cities_left: int = 4
    roads_left: int = 15


@dataclass(frozen=True)
class PrivatePlayerState:
    player_id: int
    resources: ResourceHand = field(default_factory=empty_hand)
    dev_cards_in_hand: dict[DevCardType, int] = field(default_factory=dict)
    new_dev_cards_in_hand: dict[DevCardType, int] = field(default_factory=dict)
    hidden_vp_cards: int = 0

    def canonical(self) -> "PrivatePlayerState":
        """Return a version with normalized resources and explicit dev-card counts."""
        dev_counts = {card_type: 0 for card_type in DevCardType}
        dev_counts.update({card_type: int(count) for card_type, count in self.dev_cards_in_hand.items()})
        new_dev_counts = {card_type: 0 for card_type in DevCardType}
        new_dev_counts.update({card_type: int(count) for card_type, count in self.new_dev_cards_in_hand.items()})
        return PrivatePlayerState(
            player_id=self.player_id,
            resources=normalize_hand(self.resources),
            dev_cards_in_hand=dev_counts,
            new_dev_cards_in_hand=new_dev_counts,
            hidden_vp_cards=int(self.hidden_vp_cards),
        )


@dataclass(frozen=True)
class ExactGameState:
    """
    Full hidden-information state for the exact solver foundation.

    This is intentionally a data container with no game logic. Rule transitions
    live in `rules.py`.
    """

    board: CatanBoard
    public_players: tuple[PublicPlayerState, ...]
    private_players: tuple[PrivatePlayerState, ...]
    current_player: int
    phase: TurnPhase
    robber_hex: tuple[int, int]
    bank_resources: ResourceHand
    dev_deck: tuple[DevCardType, ...]
    pending_trade: Optional[PendingTrade] = None
    turn_number: int = 0
    setup_step: int = 0
    pending_setup_vertex: Optional[tuple[float, float]] = None
    pending_discarders: tuple[int, ...] = field(default_factory=tuple)
    free_roads_remaining: int = 0
    dev_card_played_this_turn: bool = False
    dice_rolled_this_turn: bool = False
    last_roll: Optional[int] = None
    winner_id: Optional[int] = None


def make_bank_resources() -> ResourceHand:
    """Return a fresh resource bank for a standard 4-player base game."""
    return {resource: RESOURCE_BANK_START for resource in playable_resources()}


def make_dev_deck(seed: Optional[int] = None) -> tuple[DevCardType, ...]:
    """Return a shuffled base-game development deck."""
    deck = [
        card_type
        for card_type, count in DEV_CARD_DISTRIBUTION.items()
        for _ in range(count)
    ]
    rng = random.Random(seed)
    rng.shuffle(deck)
    return tuple(deck)


def make_initial_state(
    board: CatanBoard,
    n_players: int = 4,
    seed: Optional[int] = None,
) -> ExactGameState:
    """Create an empty exact state before setup placements are applied."""
    if n_players != 4:
        raise ValueError("The full solver foundation currently assumes exactly 4 players.")

    public_players = tuple(PublicPlayerState(player_id=player_id) for player_id in range(n_players))
    private_players = tuple(
        PrivatePlayerState(
            player_id=player_id,
            resources=empty_hand(),
            dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
            new_dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
            hidden_vp_cards=0,
        )
        for player_id in range(n_players)
    )
    return ExactGameState(
        board=board,
        public_players=public_players,
        private_players=private_players,
        current_player=0,
        phase=TurnPhase.SETUP,
        robber_hex=board.robber_start,
        bank_resources=make_bank_resources(),
        dev_deck=make_dev_deck(seed),
    )
