"""Action definitions for the exact 4-player Catan engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..board.board import Resource
from .state import MaritimeTrade, ResourceHand, TradeOffer


class ActionType(str, Enum):
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    ROLL = "roll"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_KNIGHT = "play_knight"
    PLAY_MONOPOLY = "play_monopoly"
    PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
    PLAY_ROAD_BUILDING = "play_road_building"
    OFFER_TRADE = "offer_trade"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"
    MARITIME_TRADE = "maritime_trade"
    END_TURN = "end_turn"
    DECLARE_VICTORY = "declare_victory"


@dataclass(frozen=True)
class RobberMove:
    target_hex: tuple[int, int]
    victim_id: Optional[int] = None


@dataclass(frozen=True)
class RollSpec:
    value: Optional[int] = None


@dataclass(frozen=True)
class DiscardSpec:
    player_id: int
    resources: ResourceHand


@dataclass(frozen=True)
class YearOfPlentySpec:
    resources: tuple[Resource, Resource]


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    payload: object = None


def make_roll(value: Optional[int] = None) -> Action:
    return Action(ActionType.ROLL, RollSpec(value=value))


def make_discard(player_id: int, resources: ResourceHand) -> Action:
    return Action(ActionType.DISCARD, DiscardSpec(player_id=player_id, resources=resources))


def make_setup_settlement(vertex: tuple[float, float]) -> Action:
    return Action(ActionType.SETUP_SETTLEMENT, payload=vertex)


def make_setup_road(edge: frozenset) -> Action:
    return Action(ActionType.SETUP_ROAD, payload=edge)


def make_build_road(edge: frozenset) -> Action:
    return Action(ActionType.BUILD_ROAD, payload=edge)


def make_build_settlement(vertex: tuple[float, float]) -> Action:
    return Action(ActionType.BUILD_SETTLEMENT, payload=vertex)


def make_build_city(vertex: tuple[float, float]) -> Action:
    return Action(ActionType.BUILD_CITY, payload=vertex)


def make_move_robber(target_hex: tuple[int, int], victim_id: Optional[int] = None) -> Action:
    return Action(ActionType.MOVE_ROBBER, payload=RobberMove(target_hex=target_hex, victim_id=victim_id))


def make_play_knight(target_hex: tuple[int, int], victim_id: Optional[int] = None) -> Action:
    return Action(ActionType.PLAY_KNIGHT, payload=RobberMove(target_hex=target_hex, victim_id=victim_id))


def make_play_monopoly(resource: Resource) -> Action:
    return Action(ActionType.PLAY_MONOPOLY, payload=resource)


def make_play_year_of_plenty(resource_a: Resource, resource_b: Resource) -> Action:
    return Action(ActionType.PLAY_YEAR_OF_PLENTY, payload=YearOfPlentySpec(resources=(resource_a, resource_b)))


def make_offer_trade(offer: TradeOffer) -> Action:
    return Action(ActionType.OFFER_TRADE, payload=offer)


def make_accept_trade() -> Action:
    return Action(ActionType.ACCEPT_TRADE)


def make_reject_trade() -> Action:
    return Action(ActionType.REJECT_TRADE)


def make_maritime_trade(trade: MaritimeTrade) -> Action:
    return Action(ActionType.MARITIME_TRADE, payload=trade)
