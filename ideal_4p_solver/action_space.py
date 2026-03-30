"""Hierarchical action definitions for an ideal 4-player solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .state import DevCardType, HexId, PlayerId, Resource, ResourceCounts, TradeOffer


class ActionType(str, Enum):
    ROLL = "roll"
    END_TURN = "end_turn"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    STEAL = "steal"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_DEV_CARD = "play_dev_card"
    MARITIME_TRADE = "maritime_trade"
    OFFER_TRADE = "offer_trade"
    ACCEPT_TRADE = "accept_trade"
    REJECT_TRADE = "reject_trade"


@dataclass(frozen=True)
class MaritimeTrade:
    give_resource: Resource
    give_count: int
    receive_resource: Resource


@dataclass(frozen=True)
class DevCardPlay:
    card_type: DevCardType
    target_hex: HexId | None = None
    target_player: PlayerId | None = None
    monopoly_resource: Resource | None = None
    free_resources: tuple[Resource, ...] = ()
    free_road_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class Action:
    """
    Atomic action used by the engine and search.

    Search should still group these by `action_type` before selecting from the
    per-type option list, otherwise trade actions dominate the branching factor.
    """

    action_type: ActionType
    payload: Any = None


def group_actions_by_type(actions: list[Action]) -> dict[ActionType, list[Action]]:
    """Partition legal actions into the action-type buckets used by search."""
    grouped: dict[ActionType, list[Action]] = {}
    for action in actions:
        grouped.setdefault(action.action_type, []).append(action)
    return grouped


def make_build_road(edge_id: int) -> Action:
    return Action(ActionType.BUILD_ROAD, payload=edge_id)


def make_build_settlement(vertex_id: int) -> Action:
    return Action(ActionType.BUILD_SETTLEMENT, payload=vertex_id)


def make_build_city(vertex_id: int) -> Action:
    return Action(ActionType.BUILD_CITY, payload=vertex_id)


def make_move_robber(hex_id: HexId) -> Action:
    return Action(ActionType.MOVE_ROBBER, payload=hex_id)


def make_steal(player_id: PlayerId) -> Action:
    return Action(ActionType.STEAL, payload=player_id)


def make_maritime_trade(
    give_resource: Resource,
    give_count: int,
    receive_resource: Resource,
) -> Action:
    return Action(
        ActionType.MARITIME_TRADE,
        payload=MaritimeTrade(
            give_resource=give_resource,
            give_count=give_count,
            receive_resource=receive_resource,
        ),
    )


def make_offer_trade(offer: TradeOffer) -> Action:
    return Action(ActionType.OFFER_TRADE, payload=offer)


def make_accept_trade() -> Action:
    return Action(ActionType.ACCEPT_TRADE)


def make_reject_trade() -> Action:
    return Action(ActionType.REJECT_TRADE)
