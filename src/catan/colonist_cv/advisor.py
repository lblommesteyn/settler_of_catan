"""Heuristic move advisor for live CV-fed states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..board.board import PortType
from ..board.board import Resource
from ..full_solver.actions import Action, ActionType, RobberMove, RollSpec
from ..full_solver.engine import ExactRulesEngine
from ..full_solver.rules import hand_size, total_victory_points
from ..full_solver.state import (
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    ExactGameState,
    MaritimeTrade,
    TradeOffer,
    TurnPhase,
    playable_resources,
)


ROLL_PROBABILITIES = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}


@dataclass(frozen=True)
class ActionAdvice:
    action: Action
    score: float
    summary: str


@dataclass(frozen=True)
class StrategyPlan:
    lean: str
    build_queue: str
    pivot: str
    hand_goal: str
    risk: str


class HeuristicActionAdvisor:
    """Rank legal actions using deterministic one-ply evaluation."""

    def __init__(self, engine: ExactRulesEngine | None = None):
        self.engine = engine or ExactRulesEngine()

    def suggest(self, state: ExactGameState, top_k: int = 5) -> list[ActionAdvice]:
        actions = self.engine.legal_actions(state)
        if not actions:
            return []

        capped_actions = self._cap_trade_candidates(state, actions)
        strategy = self.strategy_plan(state)
        scored = []
        for action in capped_actions:
            score = self._score_action(state, action) + self._strategy_bias(state, action, strategy)
            scored.append(
                ActionAdvice(
                    action=action,
                    score=score,
                    summary=self._summarize_action(state, action),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def strategy_plan(self, state: ExactGameState) -> StrategyPlan:
        player_id = state.current_player
        my_public = state.public_players[player_id]
        my_private = state.private_players[player_id].canonical()
        vp = total_victory_points(state, player_id)
        resource_pips = self._resource_pips(state, player_id)
        dominant_resource = max(resource_pips.items(), key=lambda item: item[1])[0]
        future_sites = self._future_settlement_sites(state, player_id)
        city_pips = resource_pips[Resource.ORE] + resource_pips[Resource.WHEAT]
        settlement_pips = (
            resource_pips[Resource.WOOD]
            + resource_pips[Resource.BRICK]
            + resource_pips[Resource.SHEEP]
            + resource_pips[Resource.WHEAT]
        )
        ports = my_public.ports
        legal_actions = self.engine.legal_actions(state)

        lean_key = "balanced"
        if vp >= 8 or (vp >= 7 and (my_public.has_longest_road or my_public.has_largest_army or my_private.hidden_vp_cards > 0)):
            lean_key = "closeout"
        elif ports and resource_pips[dominant_resource] >= 7.0:
            lean_key = "port_engine"
        elif city_pips >= 9.0 and resource_pips[Resource.ORE] >= 4.0 and resource_pips[Resource.WHEAT] >= 4.0:
            lean_key = "city_dev"
        elif future_sites >= 3 and (len(my_public.roads) >= 3 or settlement_pips >= 10.0):
            lean_key = "expansion"

        best_settlement = self._best_settlement_site(state, player_id)
        best_city = self._best_city_vertex(state, player_id)
        lean = self._lean_text(lean_key, dominant_resource, ports)
        build_queue = self._build_queue_text(lean_key, state, best_settlement, best_city, legal_actions, dominant_resource)
        pivot = self._pivot_text(lean_key, state, best_settlement, best_city, future_sites, dominant_resource)
        hand_goal = self._hand_goal_text(lean_key, state, best_settlement, best_city, legal_actions, dominant_resource)
        risk = self._risk_text(state, player_id, future_sites)
        return StrategyPlan(lean=lean, build_queue=build_queue, pivot=pivot, hand_goal=hand_goal, risk=risk)

    def _cap_trade_candidates(self, state: ExactGameState, actions: list[Action]) -> list[Action]:
        trades = [action for action in actions if action.action_type == ActionType.OFFER_TRADE]
        others = [action for action in actions if action.action_type != ActionType.OFFER_TRADE]
        if len(trades) <= 10:
            return others + trades
        trades.sort(key=lambda action: self._offer_trade_delta(state, action.payload), reverse=True)
        return others + trades[:10]

    def _score_action(self, state: ExactGameState, action: Action) -> float:
        player_id = state.current_player
        baseline = self._evaluate_state(state, player_id)

        if action.action_type == ActionType.OFFER_TRADE:
            return baseline + self._offer_trade_delta(state, action.payload)

        if action.action_type == ActionType.ROLL:
            expectation = 0.0
            for roll, probability in ROLL_PROBABILITIES.items():
                rolled = self.engine.apply_action(state, Action(ActionType.ROLL, RollSpec(value=roll)), seed=0)
                stabilized = self._stabilize_followup(rolled, player_id)
                expectation += probability * self._evaluate_state(stabilized, player_id)
            return expectation

        next_state = self.engine.apply_action(state, action, seed=0)
        stabilized = self._stabilize_followup(next_state, player_id)
        return self._evaluate_state(stabilized, player_id)

    def _stabilize_followup(self, state: ExactGameState, player_id: int) -> ExactGameState:
        current = state
        for _ in range(3):
            if current.phase == TurnPhase.GAME_OVER:
                return current
            if current.phase == TurnPhase.MAIN and current.free_roads_remaining > 0:
                actions = self.engine.legal_actions(current)
                if not actions:
                    return current
                current = self.engine.apply_action(
                    current,
                    max(actions, key=lambda action: self._score_followup(current, action, player_id)),
                    seed=0,
                )
                continue
            if current.phase == TurnPhase.RESOLVE_SEVEN:
                if current.pending_discarders and current.pending_discarders[0] != player_id:
                    return current
                actions = self.engine.legal_actions(current)
                if not actions:
                    return current
                current = self.engine.apply_action(
                    current,
                    max(actions, key=lambda action: self._score_followup(current, action, player_id)),
                    seed=0,
                )
                continue
            return current
        return current

    def _score_followup(self, state: ExactGameState, action: Action, player_id: int) -> float:
        next_state = self.engine.apply_action(state, action, seed=0)
        return self._evaluate_state(next_state, player_id)

    def _evaluate_state(self, state: ExactGameState, player_id: int) -> float:
        my_public = state.public_players[player_id]
        my_private = state.private_players[player_id].canonical()

        score = 100.0 * total_victory_points(state, player_id)
        score += 4.0 * self._production_pips(state, player_id)
        score += 1.6 * self._future_settlement_sites(state, player_id)
        score += 0.8 * len(my_public.roads)
        score += 3.0 * sum(my_private.resources.values())
        score += 5.0 * sum(my_private.dev_cards_in_hand.values())
        score += 4.0 * sum(my_private.new_dev_cards_in_hand.values())
        score += 10.0 * my_private.hidden_vp_cards
        if my_public.has_longest_road:
            score += 24.0
        if my_public.has_largest_army:
            score += 24.0
        if state.winner_id == player_id:
            score += 10_000.0

        opponent_pressure = 0.0
        for opponent_id, opponent_public in enumerate(state.public_players):
            if opponent_id == player_id:
                continue
            opponent_pressure += 60.0 * total_victory_points(state, opponent_id)
            opponent_pressure += 2.2 * self._production_pips(state, opponent_id)
            if opponent_public.has_longest_road:
                opponent_pressure += 10.0
            if opponent_public.has_largest_army:
                opponent_pressure += 10.0
        score -= opponent_pressure / max(len(state.public_players) - 1, 1)
        return score

    def _strategy_bias(self, state: ExactGameState, action: Action, plan: StrategyPlan) -> float:
        lean = plan.lean.lower()
        if "closeout" in lean:
            if action.action_type == ActionType.BUILD_CITY:
                return 6.0
            if action.action_type == ActionType.BUY_DEV_CARD:
                return 4.0
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                return 2.0
        if "city/dev" in lean:
            if action.action_type == ActionType.BUILD_CITY:
                return 5.0
            if action.action_type == ActionType.BUY_DEV_CARD:
                return 3.0
            if action.action_type == ActionType.MARITIME_TRADE:
                trade = action.payload
                assert isinstance(trade, MaritimeTrade)
                if trade.receive_resource in (Resource.ORE, Resource.WHEAT):
                    return 2.0
        if "expansion" in lean:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                return 6.0
            if action.action_type == ActionType.BUILD_ROAD:
                return 3.0
            if action.action_type == ActionType.PLAY_ROAD_BUILDING:
                return 4.0
        if "port engine" in lean:
            if action.action_type == ActionType.MARITIME_TRADE:
                return 4.0
            if action.action_type == ActionType.BUILD_CITY:
                return 3.0
            if action.action_type == ActionType.BUY_DEV_CARD:
                return 2.0
        if "balanced" in lean:
            if action.action_type == ActionType.BUILD_SETTLEMENT:
                return 3.0
            if action.action_type == ActionType.BUILD_CITY:
                return 2.5
            if action.action_type == ActionType.BUY_DEV_CARD:
                return 1.5
        return 0.0

    def _production_pips(self, state: ExactGameState, player_id: int) -> float:
        total = 0.0
        player = state.public_players[player_id]
        for vertex in player.settlements:
            for tile in state.board.vertex_hexes.get(vertex, []):
                if (tile.q, tile.r) != state.robber_hex:
                    total += tile.pips
        for vertex in player.cities:
            for tile in state.board.vertex_hexes.get(vertex, []):
                if (tile.q, tile.r) != state.robber_hex:
                    total += 2.0 * tile.pips
        return total

    def _resource_pips(self, state: ExactGameState, player_id: int) -> dict[Resource, float]:
        pips = {resource: 0.0 for resource in playable_resources()}
        player = state.public_players[player_id]
        city_vertices = set(player.cities)
        for vertex in set(player.settlements) | city_vertices:
            multiplier = 2.0 if vertex in city_vertices else 1.0
            for tile in state.board.vertex_hexes.get(vertex, []):
                if tile.resource == Resource.DESERT or (tile.q, tile.r) == state.robber_hex:
                    continue
                pips[tile.resource] += multiplier * tile.pips
        return pips

    def _future_settlement_sites(self, state: ExactGameState, player_id: int) -> int:
        count = 0
        for vertex in state.board.all_vertices():
            if self.engine._is_settlement_legal(state, player_id, vertex):
                count += 1
        return count

    def _resource_weights(self, state: ExactGameState, player_id: int) -> dict[Resource, float]:
        pips = self._resource_pips(state, player_id)
        weights = {}
        for resource, production in pips.items():
            weights[resource] = 1.0 / max(production + 1.0, 1.0)
        weights[Resource.ORE] += 0.35
        weights[Resource.WHEAT] += 0.25
        return weights

    def _best_settlement_site(self, state: ExactGameState, player_id: int) -> Optional[tuple[float, float]]:
        legal = [vertex for vertex in state.board.all_vertices() if self.engine._is_settlement_legal(state, player_id, vertex)]
        if not legal:
            return None
        return max(legal, key=state.board.pip_count)

    def _best_city_vertex(self, state: ExactGameState, player_id: int) -> Optional[tuple[float, float]]:
        settlements = tuple(state.public_players[player_id].settlements)
        if not settlements:
            return None
        return max(settlements, key=state.board.pip_count)

    def _lean_text(self, lean_key: str, dominant_resource: Resource, ports: tuple[PortType, ...]) -> str:
        if lean_key == "closeout":
            return "Closeout. Convert tempo into the cleanest VP line and avoid side quests."
        if lean_key == "city_dev":
            return "City/Dev Engine. Lean on ore+wheat volume, upgrade fast, and pressure with devs."
        if lean_key == "expansion":
            return "Expansion. Own the board space first, then backfill cities after the lanes are secured."
        if lean_key == "port_engine":
            port_text = self._ports_text(ports)
            return (
                f"Port Engine. Use {port_text} to turn excess {dominant_resource.value} into the cards your build queue needs."
            )
        return "Balanced Tempo. Stay flexible and spend into the fastest efficient build each turn."

    def _build_queue_text(
        self,
        lean_key: str,
        state: ExactGameState,
        best_settlement: Optional[tuple[float, float]],
        best_city: Optional[tuple[float, float]],
        legal_actions: list[Action],
        dominant_resource: Resource,
    ) -> str:
        can_settle = any(action.action_type == ActionType.BUILD_SETTLEMENT for action in legal_actions)
        can_city = any(action.action_type == ActionType.BUILD_CITY for action in legal_actions)
        can_dev = any(action.action_type == ActionType.BUY_DEV_CARD for action in legal_actions)
        can_trade = any(action.action_type == ActionType.MARITIME_TRADE for action in legal_actions)

        if lean_key == "closeout":
            if can_city and best_city is not None:
                return f"Queue: city {self._vertex_text(state, best_city)} -> dev or settlement for the 10th point."
            if can_dev:
                return "Queue: dev card -> clean closeout point next turn."
            return "Queue: trade into the cleanest immediate VP line, then close on the next spend."

        if lean_key == "city_dev":
            if best_city is not None:
                return f"Queue: city {self._vertex_text(state, best_city)} -> dev card -> next city."
            if can_dev:
                return "Queue: dev card -> city as soon as ore+wheat aligns."
            return "Queue: convert into ore+wheat -> city -> dev pressure."

        if lean_key == "expansion":
            if best_settlement is not None:
                return f"Queue: settlement {self._vertex_text(state, best_settlement)} -> road to hold lane -> city."
            return "Queue: road tempo -> settlement if space opens -> city once lanes close."

        if lean_key == "port_engine":
            if can_trade:
                return f"Queue: port-trade surplus {dominant_resource.value} -> city/dev build -> repeat."
            return f"Queue: reach live port trades -> cash surplus {dominant_resource.value} -> city/dev."

        if can_settle and best_settlement is not None:
            return f"Queue: settlement {self._vertex_text(state, best_settlement)} -> city -> dev if blocked."
        if can_city and best_city is not None:
            return f"Queue: city {self._vertex_text(state, best_city)} -> settlement if lane stays open."
        if can_dev:
            return "Queue: dev card -> city or settlement depending which line opens first."
        return "Queue: take the cheapest live spend first, then pivot into the strongest follow-up line."

    def _pivot_text(
        self,
        lean_key: str,
        state: ExactGameState,
        best_settlement: Optional[tuple[float, float]],
        best_city: Optional[tuple[float, float]],
        future_sites: int,
        dominant_resource: Resource,
    ) -> str:
        if lean_key == "closeout":
            return "Pivot: if the clean winning point is blocked, buy devs or move the robber to break the race."
        if lean_key == "city_dev":
            if best_settlement is not None:
                return f"Pivot: if ore+wheat stalls, take the fast settlement on {self._vertex_text(state, best_settlement)}."
            return "Pivot: if ore+wheat stalls, spend down efficiently instead of floating a dead city hand."
        if lean_key == "expansion":
            if future_sites <= 1:
                return "Pivot: the lane is drying up, so stop overpaying for roads and switch into city/dev tempo."
            return "Pivot: if another player enters the lane first, city your core and abandon the expensive road race."
        if lean_key == "port_engine":
            return f"Pivot: if the port line is slow, stop banking on {dominant_resource.value} and move into core city/dev cards."
        if best_city is not None:
            return f"Pivot: if the board locks up, city {self._vertex_text(state, best_city)} instead of chasing thin settlement lines."
        return "Pivot: if the fastest queue closes, take the next efficient spend instead of forcing a low-EV lane."

    def _hand_goal_text(
        self,
        lean_key: str,
        state: ExactGameState,
        best_settlement: Optional[tuple[float, float]],
        best_city: Optional[tuple[float, float]],
        legal_actions: list[Action],
        dominant_resource: Resource,
    ) -> str:
        private = state.private_players[state.current_player].canonical()
        can_settle = any(action.action_type == ActionType.BUILD_SETTLEMENT for action in legal_actions)
        can_city = any(action.action_type == ActionType.BUILD_CITY for action in legal_actions)
        can_dev = any(action.action_type == ActionType.BUY_DEV_CARD for action in legal_actions)

        if lean_key == "closeout":
            target = CITY_COST if best_city is not None else DEV_CARD_COST
            target_name = "city" if best_city is not None else "dev card"
        elif lean_key == "city_dev":
            target = CITY_COST if best_city is not None else DEV_CARD_COST
            target_name = "city" if best_city is not None else "dev card"
        elif lean_key == "expansion":
            target = SETTLEMENT_COST if best_settlement is not None else ROAD_COST
            target_name = "settlement" if best_settlement is not None else "road"
        elif lean_key == "port_engine":
            target = CITY_COST if best_city is not None else DEV_CARD_COST
            target_name = "city" if best_city is not None else "dev card"
        else:
            if can_settle:
                return "Hand goal: you can already cash this hand into a settlement."
            if can_city:
                return "Hand goal: you can already upgrade a city this turn."
            if can_dev:
                return "Hand goal: you can already buy a dev card if the board is blocked."
            if best_settlement is not None:
                target = SETTLEMENT_COST
                target_name = "settlement"
            elif best_city is not None:
                target = CITY_COST
                target_name = "city"
            else:
                target = DEV_CARD_COST
                target_name = "dev card"

        deficit = self._cost_deficit(private.resources, target)
        if not deficit:
            if target_name == "settlement" and best_settlement is not None:
                return f"Hand goal: you already have the cards for {target_name} on {self._vertex_text(state, best_settlement)}."
            if target_name == "city" and best_city is not None:
                return f"Hand goal: you already have the cards for {target_name} on {self._vertex_text(state, best_city)}."
            return f"Hand goal: you already have the cards for a {target_name}."

        return f"Hand goal: find {self._hand_to_text(deficit)} next for a {target_name}."

    def _risk_text(self, state: ExactGameState, player_id: int, future_sites: int) -> str:
        private = state.private_players[player_id].canonical()
        my_public = state.public_players[player_id]
        if hand_size(private.resources) >= 8 and state.phase in (TurnPhase.PRE_ROLL, TurnPhase.MAIN):
            return "Risk: you are over 7 cards; spend or trade down before ending the turn."
        if state.robber_hex in {(tile.q, tile.r) for vertex in my_public.cities for tile in state.board.vertex_hexes.get(vertex, [])}:
            return "Risk: the robber is pinning one of your cities; shifting production or moving it matters."
        if future_sites <= 1 and len(my_public.roads) < 5:
            return "Risk: expansion space is drying up, so tempo matters more than perfect efficiency."
        opponent_vp = max(
            total_victory_points(state, opponent_id)
            for opponent_id in range(len(state.public_players))
            if opponent_id != player_id
        )
        if opponent_vp >= 8:
            return "Risk: an opponent is in closeout range, so preserve tempo and block obvious winning lines."
        return "Risk: no immediate emergency; stay disciplined on the main build queue."

    def _offer_trade_delta(self, state: ExactGameState, offer: TradeOffer) -> float:
        weights = self._resource_weights(state, state.current_player)
        receive_value = sum(amount * weights[resource] for resource, amount in offer.receive.items())
        give_cost = sum(amount * weights[resource] for resource, amount in offer.give.items())
        acceptance_prior = 0.45 if offer.responder is not None else 0.30
        if sum(offer.give.values()) > sum(offer.receive.values()):
            acceptance_prior += 0.10
        return acceptance_prior * 30.0 * (receive_value - give_cost)

    def _summarize_action(self, state: ExactGameState, action: Action) -> str:
        if action.action_type == ActionType.ROLL:
            return "Roll the dice and take the full expected value of the turn."
        if action.action_type == ActionType.END_TURN:
            return "Pass and preserve the current board position."
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            return self._summarize_settlement(state, action.payload)
        if action.action_type == ActionType.BUILD_CITY:
            return self._summarize_city(state, action.payload)
        if action.action_type == ActionType.BUILD_ROAD:
            return self._summarize_road(state, action.payload)
        if action.action_type == ActionType.BUY_DEV_CARD:
            return "Buy a development card to convert sheep/wheat/ore into tempo or hidden VP."
        if action.action_type == ActionType.MARITIME_TRADE:
            trade = action.payload
            assert isinstance(trade, MaritimeTrade)
            return (
                f"Trade {trade.give_count} {trade.give_resource.value} to the bank "
                f"for 1 {trade.receive_resource.value}."
            )
        if action.action_type == ActionType.OFFER_TRADE:
            offer = action.payload
            assert isinstance(offer, TradeOffer)
            target = f"to player {offer.responder}" if offer.responder is not None else "to the table"
            return f"Offer {self._hand_to_text(offer.give)} for {self._hand_to_text(offer.receive)} {target}."
        if action.action_type == ActionType.PLAY_KNIGHT:
            robber_move = action.payload
            assert isinstance(robber_move, RobberMove)
            return f"Play Knight to move the robber onto {robber_move.target_hex}."
        if action.action_type == ActionType.PLAY_MONOPOLY:
            resource = action.payload
            return f"Play Monopoly on {resource.value}."
        if action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            resources = action.payload.resources
            return f"Play Year of Plenty for {resources[0].value} and {resources[1].value}."
        if action.action_type == ActionType.PLAY_ROAD_BUILDING:
            return "Play Road Building to claim two free roads."
        if action.action_type == ActionType.DECLARE_VICTORY:
            return "Declare the win."
        return action.action_type.value.replace("_", " ").capitalize()

    def _summarize_settlement(self, state: ExactGameState, vertex: tuple[float, float]) -> str:
        pip_gain = sum(tile.pips for tile in state.board.vertex_hexes.get(vertex, []))
        resources = sorted({tile.resource.value for tile in state.board.vertex_hexes.get(vertex, []) if tile.resource != Resource.DESERT})
        return f"Build a settlement on a {pip_gain}-pip spot touching {', '.join(resources)}."

    def _summarize_city(self, state: ExactGameState, vertex: tuple[float, float]) -> str:
        pip_gain = sum(tile.pips for tile in state.board.vertex_hexes.get(vertex, []))
        return f"Upgrade the {pip_gain}-pip settlement at {vertex} into a city."

    def _summarize_road(self, state: ExactGameState, edge: frozenset) -> str:
        next_state = self.engine.apply_action(state, Action(ActionType.BUILD_ROAD, edge), seed=0)
        opened = self._future_settlement_sites(next_state, state.current_player) - self._future_settlement_sites(state, state.current_player)
        return f"Build a road that opens {opened} additional settlement site(s)."

    def _ports_text(self, ports: tuple[PortType, ...]) -> str:
        if not ports:
            return "no ports"
        labels = []
        for port in ports:
            if port == PortType.GENERIC:
                labels.append("3:1")
            else:
                labels.append(port.value.replace("2:1_", "2:1 ").replace("_", ""))
        return ", ".join(labels)

    def _vertex_text(self, state: ExactGameState, vertex: tuple[float, float]) -> str:
        tiles = sorted(
            state.board.vertex_hexes.get(vertex, []),
            key=lambda tile: (tile.number is None, -(tile.number or 0), tile.resource.value),
        )
        parts = [f"{tile.resource.value} {tile.number}" for tile in tiles if tile.resource != Resource.DESERT and tile.number is not None]
        return " / ".join(parts) if parts else str(vertex)

    def _cost_deficit(self, hand: dict[Resource, int], cost: dict[Resource, int]) -> dict[Resource, int]:
        deficit = {}
        for resource, amount in cost.items():
            missing = max(0, amount - hand.get(resource, 0))
            if missing > 0:
                deficit[resource] = missing
        return deficit

    def _hand_to_text(self, hand: dict[Resource, int]) -> str:
        chunks = [f"{count} {resource.value}" for resource, count in hand.items() if count > 0]
        return ", ".join(chunks) if chunks else "nothing"
