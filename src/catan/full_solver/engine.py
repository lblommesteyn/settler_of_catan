"""Exact playable engine for 4-player base-game Catan."""

from __future__ import annotations

import random
from dataclasses import replace

from ..board.board import Resource
from .actions import (
    Action,
    ActionType,
    DiscardSpec,
    RobberMove,
    RollSpec,
    YearOfPlentySpec,
    make_accept_trade,
    make_build_city,
    make_build_road,
    make_build_settlement,
    make_discard,
    make_move_robber,
    make_offer_trade,
    make_play_knight,
    make_play_monopoly,
    make_play_year_of_plenty,
    make_roll,
    make_setup_road,
    make_setup_settlement,
)
from .rules import (
    _add_hands,
    _hand_has_at_least,
    _subtract_hands,
    accept_pending_trade,
    apply_maritime_trade,
    hand_size,
    legal_maritime_trades,
    refresh_public_state,
    reject_pending_trade,
    resolve_resource_shortage,
    start_domestic_trade,
    total_victory_points,
)
from .state import (
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    DevCardType,
    ExactGameState,
    MaritimeTrade,
    TradeOffer,
    TurnPhase,
    empty_hand,
    normalize_hand,
    playable_resources,
)


class ExactRulesEngine:
    """Exact-rules transition engine for standard 4-player base Catan."""

    def legal_actions(self, state: ExactGameState) -> list[Action]:
        state = self._normalize_stuck_free_roads(state)

        if state.phase == TurnPhase.GAME_OVER:
            return []
        if state.phase == TurnPhase.SETUP:
            return self._setup_actions(state)
        if state.phase == TurnPhase.PENDING_TRADE:
            return self._pending_trade_actions(state)
        if state.phase == TurnPhase.RESOLVE_SEVEN:
            return self._resolve_seven_actions(state)
        if state.phase == TurnPhase.PRE_ROLL:
            return self._pre_roll_actions(state)
        if state.phase == TurnPhase.MAIN:
            return self._main_actions(state)
        return []

    def apply_action(
        self,
        state: ExactGameState,
        action: Action,
        seed: int | None = None,
    ) -> ExactGameState:
        state = self._normalize_stuck_free_roads(state)
        rng = random.Random(seed)

        if action.action_type == ActionType.DECLARE_VICTORY:
            return self._declare_victory(state)
        if state.phase == TurnPhase.SETUP:
            return self._apply_setup_action(state, action)
        if state.phase == TurnPhase.PENDING_TRADE:
            return self._apply_pending_trade_action(state, action)
        if state.phase == TurnPhase.RESOLVE_SEVEN:
            return self._apply_resolve_seven_action(state, action, rng)
        if state.phase == TurnPhase.PRE_ROLL:
            return self._apply_pre_roll_action(state, action, rng)
        if state.phase == TurnPhase.MAIN:
            return self._apply_main_action(state, action, rng)
        raise ValueError(f"Unsupported action in phase {state.phase.value}: {action.action_type.value}")

    def _setup_actions(self, state: ExactGameState) -> list[Action]:
        if state.pending_setup_vertex is None:
            occupied = self._occupied_vertices(state)
            vertices = [
                vertex
                for vertex in state.board.legal_starting_vertices()
                if self._is_setup_settlement_legal(state, vertex, occupied)
            ]
            return [make_setup_settlement(vertex) for vertex in vertices]

        settlement = state.pending_setup_vertex
        edges = []
        for neighbor in state.board.graph.vertex_neighbors[settlement]:
            edge = frozenset({settlement, neighbor})
            if edge not in self._occupied_edges(state):
                edges.append(edge)
        return [make_setup_road(edge) for edge in edges]

    def _pending_trade_actions(self, state: ExactGameState) -> list[Action]:
        pending = state.pending_trade
        if pending is None or not pending.responders_left:
            return []
        return [make_accept_trade(), Action(ActionType.REJECT_TRADE)]

    def _resolve_seven_actions(self, state: ExactGameState) -> list[Action]:
        if state.pending_discarders:
            player_id = state.pending_discarders[0]
            player = state.private_players[player_id].canonical()
            discard_n = hand_size(player.resources) // 2
            return [make_discard(player_id, resources) for resources in self._enumerate_discard_hands(player.resources, discard_n)]
        return self._robber_actions(state, for_knight=False)

    def _pre_roll_actions(self, state: ExactGameState) -> list[Action]:
        actions: list[Action] = []
        if total_victory_points(state, state.current_player) >= 10:
            actions.append(Action(ActionType.DECLARE_VICTORY))
        actions.extend(self._play_dev_actions(state))
        actions.append(make_roll())
        return actions

    def _main_actions(self, state: ExactGameState) -> list[Action]:
        actions: list[Action] = []
        if total_victory_points(state, state.current_player) >= 10:
            actions.append(Action(ActionType.DECLARE_VICTORY))

        free_edges = self._free_road_edges(state, state.current_player)
        if state.free_roads_remaining > 0 and free_edges:
            return [make_build_road(edge) for edge in free_edges] + actions

        actions.extend(self._play_dev_actions(state))
        actions.extend(make_build_road(edge) for edge in self._normal_build_road_edges(state, state.current_player))
        actions.extend(make_build_settlement(vertex) for vertex in self._normal_build_settlement_vertices(state, state.current_player))
        actions.extend(make_build_city(vertex) for vertex in self._normal_build_city_vertices(state, state.current_player))
        if self._can_buy_dev_card(state, state.current_player):
            actions.append(Action(ActionType.BUY_DEV_CARD))
        actions.extend(Action(ActionType.MARITIME_TRADE, payload=trade) for trade in legal_maritime_trades(state, state.current_player))
        actions.extend(make_offer_trade(offer) for offer in self._structured_trade_offers(state))
        actions.append(Action(ActionType.END_TURN))
        return actions

    def _apply_setup_action(self, state: ExactGameState, action: Action) -> ExactGameState:
        if action.action_type == ActionType.SETUP_SETTLEMENT:
            vertex = action.payload
            if not self._is_setup_settlement_legal(state, vertex, self._occupied_vertices(state)):
                raise ValueError("Illegal setup settlement placement.")
            public_players = list(state.public_players)
            player = public_players[state.current_player]
            if player.settlements_left <= 0:
                raise ValueError("Player has no settlements left.")
            public_players[state.current_player] = replace(
                player,
                settlements=frozenset(set(player.settlements) | {vertex}),
                settlements_left=player.settlements_left - 1,
            )
            return refresh_public_state(
                replace(
                    state,
                    public_players=tuple(public_players),
                    pending_setup_vertex=vertex,
                )
            )

        if action.action_type == ActionType.SETUP_ROAD:
            edge = action.payload
            settlement = state.pending_setup_vertex
            if settlement is None or settlement not in edge:
                raise ValueError("Setup road must be adjacent to the just-placed settlement.")
            if edge in self._occupied_edges(state):
                raise ValueError("Setup road edge is already occupied.")

            public_players = list(state.public_players)
            private_players = list(state.private_players)
            player = public_players[state.current_player]
            if player.roads_left <= 0:
                raise ValueError("Player has no roads left.")
            public_players[state.current_player] = replace(
                player,
                roads=frozenset(set(player.roads) | {edge}),
                roads_left=player.roads_left - 1,
            )

            bank_resources = normalize_hand(state.bank_resources)
            if state.setup_step >= 4:
                gains = empty_hand()
                for tile in state.board.vertex_hexes.get(settlement, []):
                    if tile.resource != Resource.DESERT:
                        gains[tile.resource] += 1
                        bank_resources[tile.resource] -= 1
                private_players[state.current_player] = replace(
                    private_players[state.current_player].canonical(),
                    resources=_add_hands(private_players[state.current_player].resources, gains),
                )

            next_step = state.setup_step + 1
            if next_step >= 8:
                next_state = replace(
                    state,
                    public_players=tuple(public_players),
                    private_players=tuple(private_players),
                    bank_resources=bank_resources,
                    pending_setup_vertex=None,
                    setup_step=next_step,
                    current_player=0,
                    phase=TurnPhase.PRE_ROLL,
                )
            else:
                next_state = replace(
                    state,
                    public_players=tuple(public_players),
                    private_players=tuple(private_players),
                    bank_resources=bank_resources,
                    pending_setup_vertex=None,
                    setup_step=next_step,
                    current_player=self._setup_order()[next_step],
                )
            return refresh_public_state(next_state)

        raise ValueError(f"Illegal setup action: {action.action_type.value}")

    def _apply_pending_trade_action(self, state: ExactGameState, action: Action) -> ExactGameState:
        pending = state.pending_trade
        if pending is None or not pending.responders_left:
            raise ValueError("No trade is pending.")
        responder_id = pending.responders_left[0]
        if action.action_type == ActionType.ACCEPT_TRADE:
            return accept_pending_trade(state, responder_id)
        if action.action_type == ActionType.REJECT_TRADE:
            return reject_pending_trade(state, responder_id)
        raise ValueError("Only accept/reject are legal while a trade is pending.")

    def _apply_resolve_seven_action(
        self,
        state: ExactGameState,
        action: Action,
        rng: random.Random,
    ) -> ExactGameState:
        if state.pending_discarders:
            if action.action_type != ActionType.DISCARD:
                raise ValueError("The next action must be a discard.")
            discard = action.payload
            assert isinstance(discard, DiscardSpec)
            current_discarder = state.pending_discarders[0]
            if discard.player_id != current_discarder:
                raise ValueError("Wrong player is discarding.")
            private_player = state.private_players[current_discarder].canonical()
            required = hand_size(private_player.resources) // 2
            resources = normalize_hand(discard.resources)
            if hand_size(resources) != required:
                raise ValueError("Incorrect number of discarded cards.")
            if not _hand_has_at_least(private_player.resources, resources):
                raise ValueError("Discard exceeds player's hand.")

            private_players = list(state.private_players)
            private_players[current_discarder] = replace(
                private_player,
                resources=_subtract_hands(private_player.resources, resources),
            )
            bank_resources = _add_hands(state.bank_resources, resources)
            return refresh_public_state(
                replace(
                    state,
                    private_players=tuple(private_players),
                    bank_resources=bank_resources,
                    pending_discarders=state.pending_discarders[1:],
                )
            )

        if action.action_type != ActionType.MOVE_ROBBER:
            raise ValueError("Robber must be moved after resolving discards.")
        robber_action = action.payload
        assert isinstance(robber_action, RobberMove)
        return self._move_robber(
            replace(state, phase=TurnPhase.MAIN, dice_rolled_this_turn=True),
            robber_action,
            rng,
        )

    def _apply_pre_roll_action(
        self,
        state: ExactGameState,
        action: Action,
        rng: random.Random,
    ) -> ExactGameState:
        if action.action_type == ActionType.ROLL:
            roll_spec = action.payload if action.payload is not None else RollSpec()
            assert isinstance(roll_spec, RollSpec)
            roll = roll_spec.value if roll_spec.value is not None else rng.randint(1, 6) + rng.randint(1, 6)
            if roll == 7:
                discarders = tuple(
                    player.player_id
                    for player in state.public_players
                    if hand_size(state.private_players[player.player_id].resources) > 7
                )
                return replace(
                    state,
                    phase=TurnPhase.RESOLVE_SEVEN,
                    pending_discarders=discarders,
                    dice_rolled_this_turn=True,
                    last_roll=roll,
                )
            return self._distribute_roll(
                replace(state, phase=TurnPhase.MAIN, dice_rolled_this_turn=True, last_roll=roll),
                roll,
            )

        if action.action_type in {
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_ROAD_BUILDING,
        }:
            return self._apply_dev_action(state, action, rng)

        raise ValueError(f"Illegal pre-roll action: {action.action_type.value}")

    def _apply_main_action(
        self,
        state: ExactGameState,
        action: Action,
        rng: random.Random,
    ) -> ExactGameState:
        state = self._normalize_stuck_free_roads(state)
        if action.action_type == ActionType.BUILD_ROAD:
            return self._build_road(state, action.payload)
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            return self._build_settlement(state, action.payload)
        if action.action_type == ActionType.BUILD_CITY:
            return self._build_city(state, action.payload)
        if action.action_type == ActionType.BUY_DEV_CARD:
            return self._buy_dev_card(state)
        if action.action_type == ActionType.MARITIME_TRADE:
            assert isinstance(action.payload, MaritimeTrade)
            return apply_maritime_trade(state, action.payload)
        if action.action_type == ActionType.OFFER_TRADE:
            assert isinstance(action.payload, TradeOffer)
            return start_domestic_trade(state, action.payload)
        if action.action_type == ActionType.END_TURN:
            return self._end_turn(state)
        if action.action_type in {
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_ROAD_BUILDING,
        }:
            return self._apply_dev_action(state, action, rng)
        raise ValueError(f"Illegal main-phase action: {action.action_type.value}")

    def _build_road(self, state: ExactGameState, edge: frozenset) -> ExactGameState:
        player_id = state.current_player
        free_build = state.free_roads_remaining > 0
        if edge not in state.board.graph.edges:
            raise ValueError("Road edge is not part of the board.")
        if edge in self._occupied_edges(state):
            raise ValueError("Road edge is already occupied.")
        if not self._is_road_legal(state, player_id, edge):
            raise ValueError("Road placement is not connected legally.")

        public_players = list(state.public_players)
        private_players = list(state.private_players)
        player_public = public_players[player_id]
        player_private = private_players[player_id].canonical()

        if player_public.roads_left <= 0:
            raise ValueError("Player has no road pieces left.")
        if not free_build and not _hand_has_at_least(player_private.resources, ROAD_COST):
            raise ValueError("Player cannot afford a road.")

        new_resources = player_private.resources if free_build else _subtract_hands(player_private.resources, ROAD_COST)
        new_bank = state.bank_resources if free_build else _add_hands(state.bank_resources, ROAD_COST)

        public_players[player_id] = replace(
            player_public,
            roads=frozenset(set(player_public.roads) | {edge}),
            roads_left=player_public.roads_left - 1,
        )
        private_players[player_id] = replace(player_private, resources=new_resources)

        next_state = replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            bank_resources=new_bank,
            free_roads_remaining=max(0, state.free_roads_remaining - 1 if free_build else 0),
        )
        return refresh_public_state(next_state)

    def _build_settlement(self, state: ExactGameState, vertex: tuple[float, float]) -> ExactGameState:
        player_id = state.current_player
        if not self._is_settlement_legal(state, player_id, vertex):
            raise ValueError("Settlement placement is illegal.")

        public_players = list(state.public_players)
        private_players = list(state.private_players)
        player_public = public_players[player_id]
        player_private = private_players[player_id].canonical()

        if player_public.settlements_left <= 0:
            raise ValueError("Player has no settlements left.")
        if not _hand_has_at_least(player_private.resources, SETTLEMENT_COST):
            raise ValueError("Player cannot afford a settlement.")

        public_players[player_id] = replace(
            player_public,
            settlements=frozenset(set(player_public.settlements) | {vertex}),
            settlements_left=player_public.settlements_left - 1,
        )
        private_players[player_id] = replace(player_private, resources=_subtract_hands(player_private.resources, SETTLEMENT_COST))
        next_state = replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            bank_resources=_add_hands(state.bank_resources, SETTLEMENT_COST),
        )
        return refresh_public_state(next_state)

    def _build_city(self, state: ExactGameState, vertex: tuple[float, float]) -> ExactGameState:
        player_id = state.current_player
        player_public = state.public_players[player_id]
        if vertex not in player_public.settlements:
            raise ValueError("City upgrade requires one of the player's settlements.")
        if player_public.cities_left <= 0:
            raise ValueError("Player has no cities left.")

        player_private = state.private_players[player_id].canonical()
        if not _hand_has_at_least(player_private.resources, CITY_COST):
            raise ValueError("Player cannot afford a city.")

        public_players = list(state.public_players)
        private_players = list(state.private_players)
        public_players[player_id] = replace(
            player_public,
            settlements=frozenset(set(player_public.settlements) - {vertex}),
            cities=frozenset(set(player_public.cities) | {vertex}),
            settlements_left=player_public.settlements_left + 1,
            cities_left=player_public.cities_left - 1,
        )
        private_players[player_id] = replace(player_private, resources=_subtract_hands(player_private.resources, CITY_COST))
        next_state = replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            bank_resources=_add_hands(state.bank_resources, CITY_COST),
        )
        return refresh_public_state(next_state)

    def _buy_dev_card(self, state: ExactGameState) -> ExactGameState:
        player_id = state.current_player
        if state.dev_deck == ():
            raise ValueError("No development cards remain.")
        player_private = state.private_players[player_id].canonical()
        if not _hand_has_at_least(player_private.resources, DEV_CARD_COST):
            raise ValueError("Player cannot afford a development card.")

        drawn = state.dev_deck[0]
        remaining_deck = state.dev_deck[1:]

        public_players = list(state.public_players)
        private_players = list(state.private_players)
        public_players[player_id] = replace(
            public_players[player_id],
            dev_cards_bought=public_players[player_id].dev_cards_bought + 1,
        )

        next_private = replace(player_private, resources=_subtract_hands(player_private.resources, DEV_CARD_COST))
        if drawn == DevCardType.VICTORY_POINT:
            next_private = replace(next_private, hidden_vp_cards=next_private.hidden_vp_cards + 1)
        else:
            fresh = dict(next_private.new_dev_cards_in_hand)
            fresh[drawn] = fresh.get(drawn, 0) + 1
            next_private = replace(next_private, new_dev_cards_in_hand=fresh)
        private_players[player_id] = next_private

        next_state = replace(
            state,
            public_players=tuple(public_players),
            private_players=tuple(private_players),
            bank_resources=_add_hands(state.bank_resources, DEV_CARD_COST),
            dev_deck=remaining_deck,
        )
        return refresh_public_state(next_state)

    def _apply_dev_action(self, state: ExactGameState, action: Action, rng: random.Random) -> ExactGameState:
        if state.dev_card_played_this_turn:
            raise ValueError("Only one development card may be played per turn.")
        player_id = state.current_player
        player_private = state.private_players[player_id].canonical()

        def consume(card_type: DevCardType):
            if player_private.dev_cards_in_hand.get(card_type, 0) <= 0:
                raise ValueError(f"Player does not have a playable {card_type.value} card.")
            private_players = list(state.private_players)
            mature = dict(player_private.dev_cards_in_hand)
            mature[card_type] -= 1
            private_players[player_id] = replace(player_private, dev_cards_in_hand=mature)
            return private_players

        if action.action_type == ActionType.PLAY_KNIGHT:
            private_players = consume(DevCardType.KNIGHT)
            public_players = list(state.public_players)
            public_players[player_id] = replace(public_players[player_id], played_knights=public_players[player_id].played_knights + 1)
            base = replace(
                state,
                private_players=tuple(private_players),
                public_players=tuple(public_players),
                dev_card_played_this_turn=True,
            )
            return refresh_public_state(self._move_robber(base, action.payload, rng))

        if action.action_type == ActionType.PLAY_MONOPOLY:
            private_players = consume(DevCardType.MONOPOLY)
            resource = action.payload
            assert isinstance(resource, Resource)
            if resource == Resource.DESERT:
                raise ValueError("Cannot monopolize desert.")
            total_taken = 0
            for other_id, other in enumerate(private_players):
                if other_id == player_id:
                    continue
                other_private = other.canonical()
                count = other_private.resources[resource]
                if count:
                    updated_other = normalize_hand(other_private.resources)
                    updated_other[resource] = 0
                    private_players[other_id] = replace(other_private, resources=updated_other)
                    total_taken += count
            updated_self = private_players[player_id].canonical()
            updated_self_resources = normalize_hand(updated_self.resources)
            updated_self_resources[resource] += total_taken
            private_players[player_id] = replace(updated_self, resources=updated_self_resources)
            return refresh_public_state(replace(state, private_players=tuple(private_players), dev_card_played_this_turn=True))

        if action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            private_players = consume(DevCardType.YEAR_OF_PLENTY)
            spec = action.payload
            assert isinstance(spec, YearOfPlentySpec)
            bank = normalize_hand(state.bank_resources)
            resources = normalize_hand(private_players[player_id].resources)
            for resource in spec.resources:
                if bank[resource] <= 0:
                    raise ValueError("Bank cannot satisfy Year of Plenty request.")
                bank[resource] -= 1
                resources[resource] += 1
            private_players[player_id] = replace(private_players[player_id].canonical(), resources=resources)
            return refresh_public_state(
                replace(state, private_players=tuple(private_players), bank_resources=bank, dev_card_played_this_turn=True)
            )

        if action.action_type == ActionType.PLAY_ROAD_BUILDING:
            consume_players = consume(DevCardType.ROAD_BUILDING)
            next_state = replace(state, private_players=tuple(consume_players), dev_card_played_this_turn=True, free_roads_remaining=2)
            return self._normalize_stuck_free_roads(refresh_public_state(next_state))

        raise ValueError(f"Unsupported development-card action: {action.action_type.value}")

    def _end_turn(self, state: ExactGameState) -> ExactGameState:
        if state.phase != TurnPhase.MAIN:
            raise ValueError("Only main phase can end the turn.")

        private_players = []
        for private in state.private_players:
            canonical = private.canonical()
            mature = dict(canonical.dev_cards_in_hand)
            fresh = dict(canonical.new_dev_cards_in_hand)
            for card_type, count in fresh.items():
                mature[card_type] = mature.get(card_type, 0) + count
            private_players.append(
                replace(
                    canonical,
                    dev_cards_in_hand=mature,
                    new_dev_cards_in_hand={card_type: 0 for card_type in DevCardType},
                )
            )

        return refresh_public_state(
            replace(
                state,
                private_players=tuple(private_players),
                current_player=(state.current_player + 1) % len(state.public_players),
                phase=TurnPhase.PRE_ROLL,
                turn_number=state.turn_number + 1,
                dev_card_played_this_turn=False,
                dice_rolled_this_turn=False,
                free_roads_remaining=0,
                last_roll=None,
            )
        )

    def _declare_victory(self, state: ExactGameState) -> ExactGameState:
        if state.phase not in {TurnPhase.PRE_ROLL, TurnPhase.MAIN}:
            raise ValueError("Victory can only be declared during the active turn.")
        if total_victory_points(state, state.current_player) < 10:
            raise ValueError("Player does not have enough victory points.")
        return replace(state, phase=TurnPhase.GAME_OVER, winner_id=state.current_player)

    def _normalize_stuck_free_roads(self, state: ExactGameState) -> ExactGameState:
        if state.free_roads_remaining <= 0:
            return state
        if self._free_road_edges(state, state.current_player):
            return state
        return replace(state, free_roads_remaining=0)

    def _setup_order(self) -> list[int]:
        return [0, 1, 2, 3, 3, 2, 1, 0]

    def _occupied_vertices(self, state: ExactGameState) -> set[tuple[float, float]]:
        occupied: set[tuple[float, float]] = set()
        for player in state.public_players:
            occupied.update(player.settlements)
            occupied.update(player.cities)
        return occupied

    def _occupied_edges(self, state: ExactGameState) -> set[frozenset]:
        occupied: set[frozenset] = set()
        for player in state.public_players:
            occupied.update(player.roads)
        return occupied

    def _opponent_vertices(self, state: ExactGameState, player_id: int) -> set[tuple[float, float]]:
        vertices: set[tuple[float, float]] = set()
        for player in state.public_players:
            if player.player_id == player_id:
                continue
            vertices.update(player.settlements)
            vertices.update(player.cities)
        return vertices

    def _is_setup_settlement_legal(
        self,
        state: ExactGameState,
        vertex: tuple[float, float],
        occupied: set[tuple[float, float]],
    ) -> bool:
        if vertex in occupied:
            return False
        for neighbor in state.board.graph.vertex_neighbors[vertex]:
            if neighbor in occupied:
                return False
        return bool(state.board.vertex_hexes.get(vertex))

    def _is_road_legal(self, state: ExactGameState, player_id: int, edge: frozenset) -> bool:
        v1, v2 = tuple(edge)
        player = state.public_players[player_id]
        own_vertices = set(player.settlements) | set(player.cities)
        opponent_vertices = self._opponent_vertices(state, player_id)

        for endpoint in (v1, v2):
            if endpoint in own_vertices:
                return True
            if endpoint in opponent_vertices:
                continue
            for own_road in player.roads:
                if endpoint in own_road:
                    return True
        return False

    def _is_settlement_legal(self, state: ExactGameState, player_id: int, vertex: tuple[float, float]) -> bool:
        if vertex in self._occupied_vertices(state):
            return False
        for neighbor in state.board.graph.vertex_neighbors[vertex]:
            if neighbor in self._occupied_vertices(state):
                return False
        player = state.public_players[player_id]
        return any(vertex in road for road in player.roads)

    def _normal_build_road_edges(self, state: ExactGameState, player_id: int) -> list[frozenset]:
        player = state.public_players[player_id]
        if player.roads_left <= 0 or not _hand_has_at_least(state.private_players[player_id].resources, ROAD_COST):
            return []
        return self._free_road_edges(state, player_id)

    def _free_road_edges(self, state: ExactGameState, player_id: int) -> list[frozenset]:
        legal: list[frozenset] = []
        occupied = self._occupied_edges(state)
        for edge in state.board.graph.edges:
            if edge in occupied:
                continue
            if self._is_road_legal(state, player_id, edge):
                legal.append(edge)
        return legal

    def _normal_build_settlement_vertices(self, state: ExactGameState, player_id: int) -> list[tuple[float, float]]:
        player = state.public_players[player_id]
        if player.settlements_left <= 0 or not _hand_has_at_least(state.private_players[player_id].resources, SETTLEMENT_COST):
            return []
        return [vertex for vertex in state.board.all_vertices() if self._is_settlement_legal(state, player_id, vertex)]

    def _normal_build_city_vertices(self, state: ExactGameState, player_id: int) -> list[tuple[float, float]]:
        player = state.public_players[player_id]
        if player.cities_left <= 0 or not _hand_has_at_least(state.private_players[player_id].resources, CITY_COST):
            return []
        return sorted(player.settlements)

    def _can_buy_dev_card(self, state: ExactGameState, player_id: int) -> bool:
        return bool(state.dev_deck) and _hand_has_at_least(state.private_players[player_id].resources, DEV_CARD_COST)

    def _play_dev_actions(self, state: ExactGameState) -> list[Action]:
        if state.dev_card_played_this_turn or state.free_roads_remaining > 0:
            return []
        player_id = state.current_player
        player = state.private_players[player_id].canonical()
        actions: list[Action] = []

        if player.dev_cards_in_hand.get(DevCardType.KNIGHT, 0) > 0:
            actions.extend(self._robber_actions(state, for_knight=True))
        if player.dev_cards_in_hand.get(DevCardType.MONOPOLY, 0) > 0:
            actions.extend(make_play_monopoly(resource) for resource in playable_resources())
        if player.dev_cards_in_hand.get(DevCardType.YEAR_OF_PLENTY, 0) > 0:
            available = [resource for resource in playable_resources() if state.bank_resources[resource] > 0]
            for i, resource_a in enumerate(available):
                for resource_b in available[i:]:
                    if resource_a == resource_b and state.bank_resources[resource_a] < 2:
                        continue
                    actions.append(make_play_year_of_plenty(resource_a, resource_b))
        if player.dev_cards_in_hand.get(DevCardType.ROAD_BUILDING, 0) > 0 and self._free_road_edges(state, player_id):
            actions.append(Action(ActionType.PLAY_ROAD_BUILDING))
        return actions

    def _robber_actions(self, state: ExactGameState, for_knight: bool) -> list[Action]:
        actions: list[Action] = []
        for target_hex in sorted(state.board.tiles):
            if target_hex == state.robber_hex:
                continue
            victims = self._robber_victims(state, state.current_player, target_hex)
            if victims:
                for victim_id in victims:
                    actions.append(make_play_knight(target_hex, victim_id) if for_knight else make_move_robber(target_hex, victim_id))
            else:
                actions.append(make_play_knight(target_hex, None) if for_knight else make_move_robber(target_hex, None))
        return actions

    def _robber_victims(self, state: ExactGameState, acting_player: int, target_hex: tuple[int, int]) -> list[int]:
        victims = set()
        for player in state.public_players:
            if player.player_id == acting_player:
                continue
            vertices = set(player.settlements) | set(player.cities)
            for vertex in vertices:
                for tile in state.board.vertex_hexes.get(vertex, []):
                    if (tile.q, tile.r) == target_hex:
                        victims.add(player.player_id)
                        break
        return sorted(victims)

    def _move_robber(self, state: ExactGameState, robber_move: RobberMove, rng: random.Random) -> ExactGameState:
        if robber_move.target_hex == state.robber_hex:
            raise ValueError("Robber must move to a different hex.")
        victims = self._robber_victims(state, state.current_player, robber_move.target_hex)
        if robber_move.victim_id is None and victims:
            raise ValueError("Robber move requires selecting a victim.")
        if robber_move.victim_id is not None and robber_move.victim_id not in victims:
            raise ValueError("Selected victim is not adjacent to the robber hex.")

        private_players = list(state.private_players)
        if robber_move.victim_id is not None:
            victim = private_players[robber_move.victim_id].canonical()
            victim_cards = [resource for resource, count in victim.resources.items() for _ in range(count)]
            if victim_cards:
                stolen = rng.choice(victim_cards)
                victim_resources = normalize_hand(victim.resources)
                victim_resources[stolen] -= 1
                private_players[robber_move.victim_id] = replace(victim, resources=victim_resources)

                actor = private_players[state.current_player].canonical()
                actor_resources = normalize_hand(actor.resources)
                actor_resources[stolen] += 1
                private_players[state.current_player] = replace(actor, resources=actor_resources)

        return refresh_public_state(
            replace(
                state,
                private_players=tuple(private_players),
                robber_hex=robber_move.target_hex,
            )
        )

    def _distribute_roll(self, state: ExactGameState, roll: int) -> ExactGameState:
        demands = {player.player_id: empty_hand() for player in state.public_players}
        for player in state.public_players:
            for vertex in player.settlements:
                for tile in state.board.vertex_hexes.get(vertex, []):
                    if tile.number == roll and (tile.q, tile.r) != state.robber_hex and tile.resource != Resource.DESERT:
                        demands[player.player_id][tile.resource] += 1
            for vertex in player.cities:
                for tile in state.board.vertex_hexes.get(vertex, []):
                    if tile.number == roll and (tile.q, tile.r) != state.robber_hex and tile.resource != Resource.DESERT:
                        demands[player.player_id][tile.resource] += 2

        payouts, bank_resources = resolve_resource_shortage(state.bank_resources, demands)
        private_players = list(state.private_players)
        for player_id, payout in payouts.items():
            if hand_size(payout) == 0:
                continue
            private_players[player_id] = replace(
                private_players[player_id].canonical(),
                resources=_add_hands(private_players[player_id].resources, payout),
            )
        return refresh_public_state(replace(state, private_players=tuple(private_players), bank_resources=bank_resources))

    def _enumerate_discard_hands(self, hand: dict[Resource, int], discard_n: int) -> list[dict[Resource, int]]:
        resources = list(playable_resources())
        normalized = normalize_hand(hand)
        results: list[dict[Resource, int]] = []

        def rec(index: int, remaining: int, current: dict[Resource, int]) -> None:
            if index == len(resources):
                if remaining == 0:
                    results.append(normalize_hand(current))
                return
            resource = resources[index]
            max_take = min(normalized[resource], remaining)
            for take in range(max_take + 1):
                current[resource] = take
                rec(index + 1, remaining - take, current)
            current.pop(resource, None)

        rec(0, discard_n, {})
        return results

    def _structured_trade_offers(self, state: ExactGameState) -> list[TradeOffer]:
        player_id = state.current_player
        player = state.private_players[player_id].canonical()
        offers: list[TradeOffer] = []
        responders = [public.player_id for public in state.public_players if public.player_id != player_id]

        for give_resource in playable_resources():
            if player.resources[give_resource] <= 0:
                continue
            for give_count in range(1, min(2, player.resources[give_resource]) + 1):
                for receive_resource in playable_resources():
                    if receive_resource == give_resource:
                        continue
                    give = empty_hand()
                    receive = empty_hand()
                    give[give_resource] = give_count
                    receive[receive_resource] = 1
                    for responder in responders + [None]:
                        offers.append(
                            TradeOffer(
                                offerer=player_id,
                                responder=responder,
                                give=give,
                                receive=receive,
                            )
                        )
        return offers
