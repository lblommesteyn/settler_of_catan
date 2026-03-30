"""Exact rule helpers for the full 4-player Catan solver foundation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Optional

from ..board.board import PORT_RESOURCE, CatanBoard, PortType, Resource
from .state import (
    ExactGameState,
    LARGEST_ARMY_MIN_KNIGHTS,
    LONGEST_ROAD_MIN_LENGTH,
    MaritimeTrade,
    PendingTrade,
    ResourceHand,
    TradeOffer,
    TurnPhase,
    empty_hand,
    normalize_hand,
    playable_resources,
)


def hand_size(hand: ResourceHand) -> int:
    """Total number of resource cards in hand."""
    return sum(normalize_hand(hand).values())


def _hand_has_at_least(hand: ResourceHand, needed: ResourceHand) -> bool:
    left = normalize_hand(hand)
    right = normalize_hand(needed)
    return all(left[resource] >= right[resource] for resource in playable_resources())


def _add_hands(left: ResourceHand, right: ResourceHand) -> ResourceHand:
    left_norm = normalize_hand(left)
    right_norm = normalize_hand(right)
    return {
        resource: left_norm[resource] + right_norm[resource]
        for resource in playable_resources()
    }


def _subtract_hands(left: ResourceHand, right: ResourceHand) -> ResourceHand:
    left_norm = normalize_hand(left)
    right_norm = normalize_hand(right)
    result = {
        resource: left_norm[resource] - right_norm[resource]
        for resource in playable_resources()
    }
    for resource, amount in result.items():
        if amount < 0:
            raise ValueError(f"Resource hand went negative for {resource.value}: {amount}")
    return result


def validate_trade_offer(offer: TradeOffer) -> list[str]:
    """
    Validate a domestic trade against base-game trade constraints.

    Returns a list of rule violations; an empty list means the schema is valid.
    """
    canonical = offer.canonical()
    errors: list[str] = []

    if canonical.responder is not None and canonical.responder == canonical.offerer:
        errors.append("Offerer and responder must be different players.")

    give_total = hand_size(canonical.give)
    receive_total = hand_size(canonical.receive)
    if give_total <= 0:
        errors.append("Domestic trades must give at least one resource.")
    if receive_total <= 0:
        errors.append("Domestic trades must receive at least one resource.")

    overlapping = [
        resource.value
        for resource in playable_resources()
        if canonical.give[resource] > 0 and canonical.receive[resource] > 0
    ]
    if overlapping:
        errors.append(
            "Domestic trades may not exchange like resources: "
            + ", ".join(sorted(overlapping))
        )

    return errors


def domestic_trade_errors(state: ExactGameState, offer: TradeOffer) -> list[str]:
    """Validate that a domestic offer is legal in the current exact state."""
    errors = validate_trade_offer(offer)
    canonical = offer.canonical()

    if state.phase != TurnPhase.MAIN:
        errors.append("Domestic trades are only legal during the main phase.")
    if canonical.offerer != state.current_player:
        errors.append("Only the current player may offer a domestic trade.")
    if state.pending_trade is not None:
        errors.append("A trade is already pending.")
    if not 0 <= canonical.offerer < len(state.public_players):
        errors.append("Offerer is not a valid player index.")
    if canonical.responder is not None and not 0 <= canonical.responder < len(state.public_players):
        errors.append("Responder is not a valid player index.")

    if not errors:
        offerer_private = state.private_players[canonical.offerer].canonical()
        if not _hand_has_at_least(offerer_private.resources, canonical.give):
            errors.append("Offerer does not hold the promised resources.")
    return errors


def start_domestic_trade(state: ExactGameState, offer: TradeOffer) -> ExactGameState:
    """Start a pending domestic trade."""
    errors = domestic_trade_errors(state, offer)
    if errors:
        raise ValueError("; ".join(errors))

    canonical = offer.canonical()
    if canonical.responder is None:
        responders_left = tuple(
            player.player_id
            for player in state.public_players
            if player.player_id != canonical.offerer
        )
    else:
        responders_left = (canonical.responder,)

    return replace(
        state,
        phase=TurnPhase.PENDING_TRADE,
        pending_trade=PendingTrade(offer=canonical, responders_left=responders_left),
    )


def can_accept_pending_trade(state: ExactGameState, responder_id: int) -> list[str]:
    """Return any reasons the nominated player cannot accept the current trade."""
    errors: list[str] = []
    pending = state.pending_trade
    if state.phase != TurnPhase.PENDING_TRADE or pending is None:
        errors.append("No domestic trade is pending.")
        return errors

    if responder_id not in pending.responders_left:
        errors.append("Player is not eligible to respond to the pending trade.")

    offer = pending.offer
    if offer.responder is not None and offer.responder != responder_id:
        errors.append("This pending trade is targeted at a different responder.")

    responder_private = state.private_players[responder_id].canonical()
    if not _hand_has_at_least(responder_private.resources, offer.receive):
        errors.append("Responder does not hold the requested resources.")
    return errors


def accept_pending_trade(state: ExactGameState, responder_id: int) -> ExactGameState:
    """Execute the current pending domestic trade with the chosen responder."""
    errors = can_accept_pending_trade(state, responder_id)
    if errors:
        raise ValueError("; ".join(errors))

    pending = state.pending_trade
    assert pending is not None
    offer = pending.offer

    offerer_private = state.private_players[offer.offerer].canonical()
    responder_private = state.private_players[responder_id].canonical()

    new_offerer = replace(
        offerer_private,
        resources=_add_hands(_subtract_hands(offerer_private.resources, offer.give), offer.receive),
    )
    new_responder = replace(
        responder_private,
        resources=_add_hands(_subtract_hands(responder_private.resources, offer.receive), offer.give),
    )

    private_players = list(state.private_players)
    private_players[offer.offerer] = new_offerer
    private_players[responder_id] = new_responder

    refreshed = replace(
        state,
        private_players=tuple(private_players),
        phase=TurnPhase.MAIN,
        pending_trade=None,
    )
    return refresh_public_state(refreshed)


def reject_pending_trade(state: ExactGameState, responder_id: int) -> ExactGameState:
    """Record a rejection for one responder and clear the trade if nobody remains."""
    pending = state.pending_trade
    if state.phase != TurnPhase.PENDING_TRADE or pending is None:
        raise ValueError("No domestic trade is pending.")
    if responder_id not in pending.responders_left:
        raise ValueError("Player is not eligible to reject the pending trade.")

    remaining = tuple(pid for pid in pending.responders_left if pid != responder_id)
    if remaining:
        return replace(state, pending_trade=replace(pending, responders_left=remaining))
    return replace(state, phase=TurnPhase.MAIN, pending_trade=None)


def maritime_trade_ratio(ports: tuple[PortType, ...], give_resource: Resource) -> int:
    """Best maritime ratio available to a player for a specific resource."""
    if give_resource == Resource.DESERT:
        raise ValueError("Desert is not a tradable resource.")

    best_ratio = 4
    for port in ports:
        if port == PortType.GENERIC:
            best_ratio = min(best_ratio, 3)
        elif PORT_RESOURCE[port] == give_resource:
            best_ratio = min(best_ratio, 2)
    return best_ratio


def legal_maritime_trades(state: ExactGameState, player_id: int) -> list[MaritimeTrade]:
    """Enumerate all legal maritime trades for one player."""
    if state.phase != TurnPhase.MAIN:
        return []

    public_player = state.public_players[player_id]
    private_player = state.private_players[player_id].canonical()
    trades: list[MaritimeTrade] = []

    for give_resource in playable_resources():
        ratio = maritime_trade_ratio(public_player.ports, give_resource)
        if private_player.resources[give_resource] < ratio:
            continue

        for receive_resource in playable_resources():
            if receive_resource == give_resource:
                continue
            if state.bank_resources[receive_resource] <= 0:
                continue
            trades.append(
                MaritimeTrade(
                    player_id=player_id,
                    give_resource=give_resource,
                    give_count=ratio,
                    receive_resource=receive_resource,
                )
            )
    return trades


def apply_maritime_trade(state: ExactGameState, trade: MaritimeTrade) -> ExactGameState:
    """Apply one legal maritime trade."""
    if state.phase != TurnPhase.MAIN:
        raise ValueError("Maritime trades are only legal during the main phase.")
    if trade.player_id != state.current_player:
        raise ValueError("Only the current player may perform a maritime trade.")
    if trade.receive_resource == trade.give_resource:
        raise ValueError("Maritime trades must exchange different resources.")

    legal = set(legal_maritime_trades(state, trade.player_id))
    if trade not in legal:
        raise ValueError("Maritime trade is not legal in the current state.")

    player_private = state.private_players[trade.player_id].canonical()
    updated_resources = normalize_hand(player_private.resources)
    updated_resources[trade.give_resource] -= trade.give_count
    updated_resources[trade.receive_resource] += 1

    updated_bank = normalize_hand(state.bank_resources)
    updated_bank[trade.give_resource] += trade.give_count
    updated_bank[trade.receive_resource] -= 1

    private_players = list(state.private_players)
    private_players[trade.player_id] = replace(player_private, resources=updated_resources)

    refreshed = replace(
        state,
        private_players=tuple(private_players),
        bank_resources=updated_bank,
    )
    return refresh_public_state(refreshed)


def resolve_resource_shortage(
    bank_resources: ResourceHand,
    player_demands: dict[int, ResourceHand],
) -> tuple[dict[int, ResourceHand], ResourceHand]:
    """
    Apply the exact base-game shortage rule to a set of resource demands.

    For each resource type independently:
    - if bank can satisfy all demand, distribute normally
    - if only one player is entitled, that player gets all available cards
    - if multiple players are entitled and the bank is short, nobody gets that
      resource this turn
    """
    payouts = {player_id: empty_hand() for player_id in player_demands}
    updated_bank = normalize_hand(bank_resources)
    demands = {player_id: normalize_hand(hand) for player_id, hand in player_demands.items()}

    for resource in playable_resources():
        recipients = [player_id for player_id, hand in demands.items() if hand[resource] > 0]
        if not recipients:
            continue

        total_demand = sum(demands[player_id][resource] for player_id in recipients)
        available = updated_bank[resource]

        if total_demand <= available:
            for player_id in recipients:
                payouts[player_id][resource] = demands[player_id][resource]
            updated_bank[resource] -= total_demand
        elif len(recipients) == 1:
            only_player = recipients[0]
            payouts[only_player][resource] = available
            updated_bank[resource] = 0

    return payouts, updated_bank


def exact_longest_road_length(
    board: CatanBoard,
    player_roads: frozenset[frozenset],
    blocked_vertices: Optional[set[tuple[float, float]]] = None,
) -> int:
    """
    Compute the exact longest road length for one player.

    The search is over edge-simple trails, which correctly handles cycles. A
    vertex occupied by an opponent interrupts traversal but may still serve as
    an endpoint of the road.
    """
    blocked = blocked_vertices or set()
    adjacency: dict[tuple[float, float], list[tuple[frozenset, tuple[float, float]]]] = defaultdict(list)

    for edge in player_roads:
        if edge not in board.graph.edges:
            raise ValueError("Road edge is not part of the board graph.")
        v1, v2 = board.graph.edges[edge]
        adjacency[v1].append((edge, v2))
        adjacency[v2].append((edge, v1))

    best = 0

    def dfs(vertex: tuple[float, float], used_edges: set[frozenset]) -> None:
        nonlocal best
        best = max(best, len(used_edges))
        if vertex in blocked and used_edges:
            return
        for edge, neighbor in adjacency.get(vertex, []):
            if edge in used_edges:
                continue
            used_edges.add(edge)
            dfs(neighbor, used_edges)
            used_edges.remove(edge)

    for start_vertex in adjacency:
        dfs(start_vertex, set())
    return best


def _player_blocked_vertices(
    state: ExactGameState,
    player_id: int,
) -> set[tuple[float, float]]:
    blocked: set[tuple[float, float]] = set()
    for public_player in state.public_players:
        if public_player.player_id == player_id:
            continue
        blocked.update(public_player.settlements)
        blocked.update(public_player.cities)
    return blocked


def _longest_road_holder(
    previous_holder: Optional[int],
    road_lengths: dict[int, int],
) -> Optional[int]:
    eligible = {player_id: length for player_id, length in road_lengths.items() if length >= LONGEST_ROAD_MIN_LENGTH}
    if not eligible:
        return None

    best_length = max(eligible.values())
    leaders = [player_id for player_id, length in eligible.items() if length == best_length]
    if len(leaders) == 1:
        return leaders[0]
    if previous_holder in leaders:
        return previous_holder
    return None


def _largest_army_holder(
    previous_holder: Optional[int],
    played_knights: dict[int, int],
) -> Optional[int]:
    eligible = {player_id: count for player_id, count in played_knights.items() if count >= LARGEST_ARMY_MIN_KNIGHTS}
    if not eligible:
        return None

    best_count = max(eligible.values())
    leaders = [player_id for player_id, count in eligible.items() if count == best_count]
    if len(leaders) == 1:
        return leaders[0]
    if previous_holder in leaders:
        return previous_holder
    return None


def _compute_player_ports(board: CatanBoard, player) -> tuple[PortType, ...]:
    port_types = {
        board.get_port(vertex).port_type
        for vertex in set(player.settlements) | set(player.cities)
        if board.get_port(vertex) is not None
    }
    return tuple(sorted(port_types, key=lambda port: port.value))


def refresh_public_state(state: ExactGameState) -> ExactGameState:
    """
    Recompute public hand sizes, ports, visible VP, and special-card holders.

    This keeps public metadata coherent after resource or road updates while the
    rest of the engine is still under construction.
    """
    previous_longest_holder = next(
        (player.player_id for player in state.public_players if player.has_longest_road),
        None,
    )
    previous_largest_holder = next(
        (player.player_id for player in state.public_players if player.has_largest_army),
        None,
    )

    road_lengths = {
        player.player_id: exact_longest_road_length(
            state.board,
            player.roads,
            blocked_vertices=_player_blocked_vertices(state, player.player_id),
        )
        for player in state.public_players
    }
    knight_counts = {player.player_id: player.played_knights for player in state.public_players}

    longest_holder = _longest_road_holder(previous_longest_holder, road_lengths)
    largest_holder = _largest_army_holder(previous_largest_holder, knight_counts)

    refreshed_public = []
    for player in state.public_players:
        private_player = state.private_players[player.player_id].canonical()
        visible_vp = (
            len(player.settlements)
            + 2 * len(player.cities)
            + (2 if player.player_id == longest_holder else 0)
            + (2 if player.player_id == largest_holder else 0)
        )
        refreshed_public.append(
            replace(
                player,
                hand_size=hand_size(private_player.resources),
                visible_vp=visible_vp,
                longest_road_length=road_lengths[player.player_id],
                has_longest_road=(player.player_id == longest_holder),
                has_largest_army=(player.player_id == largest_holder),
                ports=_compute_player_ports(state.board, player),
            )
        )

    return replace(state, public_players=tuple(refreshed_public))


def total_victory_points(state: ExactGameState, player_id: int) -> int:
    """Visible VP plus hidden victory-point development cards."""
    public_player = state.public_players[player_id]
    private_player = state.private_players[player_id].canonical()
    return public_player.visible_vp + private_player.hidden_vp_cards
