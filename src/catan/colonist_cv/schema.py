"""Mapped observation schema for a CV-driven Colonist assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..board.board import CatanBoard
from ..full_solver.state import (
    DevCardType,
    PendingTrade,
    ResourceHand,
    TurnPhase,
    empty_hand,
    normalize_hand,
)


class PlayerColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    ORANGE = "orange"
    GREEN = "green"
    WHITE = "white"

    @classmethod
    def from_value(cls, value: str) -> "PlayerColor":
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown player color: {value}")


@dataclass(frozen=True)
class PublicStructures:
    """Publicly visible board pieces for one player."""

    player_id: int
    color: Optional[PlayerColor] = None
    settlements: frozenset[tuple[float, float]] = field(default_factory=frozenset)
    cities: frozenset[tuple[float, float]] = field(default_factory=frozenset)
    roads: frozenset[frozenset] = field(default_factory=frozenset)
    visible_vp: Optional[int] = None
    played_knights: int = 0
    dev_cards_bought: int = 0


@dataclass(frozen=True)
class PrivateObservation:
    """Private cards for the point-of-view player."""

    player_id: int
    resources: ResourceHand = field(default_factory=empty_hand)
    dev_cards_in_hand: dict[DevCardType, int] = field(default_factory=dict)
    new_dev_cards_in_hand: dict[DevCardType, int] = field(default_factory=dict)
    hidden_vp_cards: int = 0

    def canonical(self) -> "PrivateObservation":
        mature = {card_type: 0 for card_type in DevCardType}
        mature.update({card_type: int(count) for card_type, count in self.dev_cards_in_hand.items()})
        fresh = {card_type: 0 for card_type in DevCardType}
        fresh.update({card_type: int(count) for card_type, count in self.new_dev_cards_in_hand.items()})
        return PrivateObservation(
            player_id=self.player_id,
            resources=normalize_hand(self.resources),
            dev_cards_in_hand=mature,
            new_dev_cards_in_hand=fresh,
            hidden_vp_cards=int(self.hidden_vp_cards),
        )


@dataclass(frozen=True)
class VisionFrameObservation:
    """
    CV-mapped snapshot of a Colonist board.

    The detector should populate only what it can read reliably. Hidden
    information for opponents can be carried forward by the tracker.
    """

    board: CatanBoard
    robber_hex: tuple[int, int]
    public_players: tuple[PublicStructures, ...]
    current_player: int
    phase: TurnPhase
    private_pov: Optional[PrivateObservation] = None
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
